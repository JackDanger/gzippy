# ONE-engine u8-flip-in-place design (P1 deliverable)

Charter: plans/engine-campaign.md. Facts base: plans/engine-u8-map.md (all vendor
citations verified first-hand). Governing memories honored: `project_faithful_unified_decoder`
(ONE ring that flips u16→u8 WIDTH in place; clean bulk u8-DIRECT; no second engine) and
`feedback_bias_guardrails` (only delete-divergent or create-match changes).

## 1. What P1's first-hand audit changed about the problem statement

The audit found the engine-internal u8 architecture **already largely landed** since the
memories were banked (2026-06-07): `marker_inflate::Block` flips width in place
(marker_inflate.rs:859-895), decodes the clean bulk u8-direct (`<false>` specialization +
`decode_clean_into_contig`), and the storage/apply layer mirrors vendor (SegmentedU16
in-place fused resolve, contiguous SegmentedU8). The REMAINING unfaithfulness is at the
**engine-graph level, not the inner loop**: gzippy-native still runs a SECOND clean engine
(`inflate::unified::Inflate` behind `StreamingInflateWrapper`) for every window-seeded chunk
and every until-exact decode (DIV-1), plus legacy alternates (DIV-3), plus two small Block
fidelity gaps (DIV-4 seam temp, DIV-5 stored-block early flips).

So the u8 rewrite = **finish the unification onto the one Block engine**, with a clean
extracted core (`WidthRing`) that makes the width state a type instead of scattered fields.

## 2. The core type: `WidthRing` (skeleton landed in P1)

`src/decompress/parallel/width_ring.rs` — the vendor window pair
(`m_window16` + `getWindow()` + `m_containsMarkerBytes` + `m_distanceToLastMarkerByte` +
`m_windowPosition`) as ONE type:

- one `Box<[u16; RING_SIZE]>` allocation; u8 view = same bytes (deflate.hpp:805-806, 890-894);
- `width(): Marker | Clean` — the selector (deflate.hpp:936-939);
- `push_literal` / `copy_backref` — width-dispatched emits with the vendor marker-counter
  rules (literal ++ / backref backward scan, deflate.hpp:1311-1317, 1379-1389);
- `should_flip(decoded_bytes)` — the EXACT arming predicate (deflate.hpp:1282-1284);
- `flip_in_place()` — conflate + tail placement + cursor re-base (deflate.hpp:1762-1784),
  returning the seam bytes as u8 so the caller can emit them WITHOUT the DIV-4 temp narrow;
- `seed_window(&[u8])` — pre-decode prime, clean from byte 0 (deflate.hpp:1750-1759);
- `drained_*` view accessors for the per-`read()` resumable drain.

P1 ships it un-wired (zero production change) with differential tests (see §6). In P2,
`Block`'s ring fields are replaced by a `WidthRing` member — a mechanical, byte-exact-gated
move; the optimized fast loops keep writing through raw pointers obtained FROM the ring
(same codegen), so the type is an ownership/contract boundary, not a perf risk.

## 3. Shape decision: in-place flip hybrid (vendor's own shape) — NOT a decode-all-then-apply rework

The charter asks for an evaluation of "rg's decode-all-then-apply shape vs the in-place flip".
Verdict from the vendor source: **these are not alternatives; rapidgzip does BOTH, layered,
and gzippy already mirrors that layering.**

- Engine layer: vendor flips in place mid-chunk and continues u8-direct in the SAME loop
  (deflate.hpp:1282-1292). There is no vendor mode that keeps decoding u16 after the window
  is resolvable. Adopting "decode the whole chunk u16, then apply" as the engine shape would
  DIVERGE from vendor and double the store traffic on the clean tail (u16 stores + a narrow
  pass vs u8 stores). REJECTED, with mechanism: vendor's existence proof runs the other way —
  its clean bulk is u8-direct, and the apply pass covers only the marker PREFIX.
- Chunk layer: when the flip never arms (marker-heavy spans — the silesia-T4 class), the whole
  chunk IS decoded u16 by the fast marker loop and resolved in ONE cache-friendly u8-LUT apply
  pass when the window arrives (DecodedData.hpp:305-391 ↔ chunk_fetcher.rs:2745-2758). That is
  the "decode-all-then-apply" the 2026-06-10 masked decomposition saw; gzippy has it, including
  the 64 KiB-LUT fused resolve+narrow and pool-side CRC. The fix for marker-heavy spans is
  marker-loop SYMBOL RATE (the u16 fast loop, marker_inflate.rs:1700-1830, + P3 arsenal on the
  unified core), not a shape change.
- ISAL build: vendor itself hands the clean tail to ISA-L at 32 KiB clean
  (GzipChunk.hpp:520-526). gzippy-isal's identical fork is FAITHFUL and stays.

## 4. Integration contracts

1. **Resumable contract** (callers in `parallel/`): unchanged — `Block::read()` keeps the
   per-call drain; `WidthRing` exposes `[drained, pos)` views for both widths. The seam call
   drains via the flip's conflate output (removes DIV-4's temp).
2. **ISA-L clean-tail handoff (gzippy-isal)**: untouched. The chunk-driver fork
   (gzip_chunk.rs:1962-2007, cfg(isal_clean_tail)) keeps returning `FlipToClean` with
   `last_32kib_window_vec()`; nothing in the WidthRing migration changes the bytes or the
   window the ISA-L tail receives. Kill-switch: the migration lands behind byte-exact gates,
   never touching `StreamingInflateWrapper`'s ISA-L variant.
3. **gzippy-native fold**: `decode_clean_into_contig` (contig-direct, DIV-2) is the SINGLE
   clean-destination contract going forward — kept under the meets-or-beats clause (one store
   pass vs vendor's ring+copy two). Window-seeded chunks (DIV-1 fix) enter the SAME contract:
   `Block.reset(seed)` → clean-from-byte-0 → `decode_clean_into_contig` with the seed bytes
   addressable for backrefs (seed lives in the ring's u8 view; first 32 KiB of backrefs read
   the ring, later ones read `chunk.data` — exactly the current post-flip fold semantics; the
   contig loop's `base[pos-d]` plus ring-window fallback already handles the seam, see
   gzip_chunk.rs:1399-1410 `data_prefix_len = 0` contract).
4. **Apply pass**: unchanged (converged): SegmentedU16 fused in-place resolve+narrow + pool CRC.

## 5. Migration steps (each landable, each byte-exact-gated)

- **M1 (this PR)**: `WidthRing` skeleton + differential tests; no wiring. Zero behavior change.
- **M2**: `Block` internals → `WidthRing` member (mechanical; fast loops take raw ptrs from the
  ring). Gate: full lib tests + silesia/model/bignasa/nasa sha differential, both builds.
  Includes DIV-4 fix (seam drain from conflate) and DIV-5 (stored-block early-flip cases,
  ported from deflate.hpp:1212-1256).
- **M3**: native window-seeded inexact chunks → Block-with-seed → contig (DIV-1 part 1), behind
  `GZIPPY_SEEDED_BLOCK=0` kill-switch for A/B. Gate: byte-exact suite + the seam-crossing nets
  (`GZIPPY_POISON_RESERVE`) + routing deletion-traps.
- **M4**: native until-exact paths → Block with an exact-stop condition (DIV-1 part 2; vendor
  analog is the wrapper's `exactUntilOffset` contract, GzipChunk.hpp:252-263 — Block gains a
  `stop_at_bit` cap checked at block boundaries, same place the driver already checks
  stop_hint). Gate: until_exact unit nets + multi-member + BGZF suites.
- **M5**: delete legacy `MarkerRing` + env fork + unused clean variants + `drain_clean_u8`
  (DIV-3/DIV-6). Gate: grep-dead + full suite.
- **M6 (P2 exit)**: masked A/B native T1/T4/T8 per charter; then P3 arsenal on the one core.

## 6. P1 skeleton tests (landed)

- flip semantics: arming predicate truth-table (both vendor conditions + non-arming cases);
  conflate places the last 32 KiB at the u8-view tail; cursor re-base; seam bytes returned.
- marker-resolution equivalence: synthetic op-streams (literal/backref mixes, marker-reaching
  distances) replayed into (a) WidthRing marker-mode → MapMarkers resolve, (b) WidthRing
  seeded clean-mode — byte-identical outputs.
- reference-model differential: WidthRing vs a trivially-correct flat history model across
  randomized generated corpora (clean-heavy, marker-heavy, flip-crossing, wrap-crossing).
- engine cross-check: existing `Block` decoding a flate2-with-dict generated corpus, seeded vs
  marker+resolve paths, against flate2 ground truth (locks the contract WidthRing must keep
  through M2).

## 7. Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| silesia-T4-class marker-heavy spans: flip never arms, whole chunk u16 | M2 must not slow the u16 fast loop (it carries these chunks) | fast loops keep raw-ptr codegen; masked A/B at M6; marker-loop knob (`GZIPPY_SLOW_MARKER_MODE`) re-run after M2 |
| window-key semantics on seeded contig (M3): backrefs read seed (ring) vs prior output (`chunk.data`) at the 32 KiB seam | wrong-byte risk, silent on zeroed pages | `GZIPPY_POISON_RESERVE` seam nets; dict-boundary differential corpus (backref exactly distance 32768 at offset 0/1); sha-pinned 5-archive suite |
| until_exact paths (M4): exact-stop on Block at non-byte-aligned boundaries / stored-block ambiguity | index-based + bgzf decode correctness | vendor stop-condition port verbatim (GzipChunk.hpp:339-345 inexact, :252-263 exact); dedicated until_exact nets before flipping the route |
| stored blocks at/after the seam (DIV-5 port) | early-flip cases interact with `uncompressed_size` resume state | port all three vendor cases with per-case unit nets (≥32 KiB stored, marker+stored crossing 32 KiB, clean stored bulk) |
| isal build regression via shared code motion | gzippy-isal is the currently-passing build | M-gates run BOTH builds' suites; isal handoff code untouched until M5's deletes (which are native-only paths) |
| `unified::Inflate` deletion breaks non-chunk callers | scan_inflate / bgzf may share it | M5 scoped to the parallel chunk graph only; routing deletion-traps (`MARKER_PIPELINE_RUNS`, `UNIFIED_INFLATE_RUNS`) updated in the same commit |

## 8. Open questions (carried to P2)

1. Does the seeded-contig fold (M3) need the ring at all, or can the seed be prepended into
   `chunk.data`'s reserve so backrefs are pure `base[pos-d]`? (Vendor seeds the ring; prepending
   deviates further but kills the seam branch. Decide on masked A/B, not taste.)
2. `Block::read`'s `n_max_to_decode` resumable cap is `usize::MAX` from the chunk driver —
   after M4, does any caller still need sub-call resumability, or can the drain move to
   block boundaries only (vendor drains per `read()` call too, so keeping it is the default)?
3. Whether the canonical (non-LUT) loop retains any caller after M5 (vendor has no such
   fallback; gzippy keeps it for non-x86 unit tests — decide whether aarch64 production uses
   the LUT path unconditionally).

---

## GATE AMENDMENTS (2026-06-10, Opus disproof gate — BINDING on M2+)

Verdict: SOUND-WITH-CHANGES. Headline + all 6 divergences verified first-hand
(flip/arming byte-identical to vendor; DIV-1 confirmed on the full-file hot path:
every window-resolved chunk takes the second engine via chunk_fetcher.rs:2564;
applyWindow-is-marker-only RECONCILES the 964/4320ms trace — on silesia the
marker region ≈ the whole chunk, which is WHY the lever is marker-loop symbol
rate, not a shape change).

1. **M2 SPLIT (required):** DIV-5 (vendor stored-block early-flip, three cases,
   deflate.hpp:1212-1256) is a BEHAVIOR change, not a mechanical refactor — it
   moves OUT of the M2 WidthRing-member migration into its own kill-switched,
   byte-exact-gated step (new M2b), so a stored-block regression cannot hide
   inside the migration commit.
2. **M4 RELABEL + CONTRACT (required before M4):** vendor's exact-stop path is
   decodeChunkWithInflateWrapper<Zlib/IsalInflateWrapper> (GzipChunk.hpp:192-265)
   — a C-FFI wrapper, NOT Block. Block-with-exact-stop is a DEVIATION justified
   solely by gzippy-native's no-C-FFI charter; label it so. Pre-registered
   contract Block must replicate from unified::Inflate (finish_decode_chunk_impl:
   890-1062): stopping-point reactions (END_OF_BLOCK / END_OF_STREAM_HEADER),
   the exact final_bit != stop_hint_bits => error assertion (gzip_chunk.rs:1062),
   footer/multi-stream (read_footer_at_current / reset_for_next_stream), and
   block-boundary recording (take_block_boundaries).
3. **§3 note (non-blocking, adopted):** vendor applyWindow iterates ONLY
   dataWithMarkers; the "marker prefix" is corpus-dependent and ≈100% of output
   on silesia-class corpora.
