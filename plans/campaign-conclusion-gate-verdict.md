# CAMPAIGN-CONCLUSION GATE — INDEPENDENT DISPROOF ADVISOR VERDICT (READ-ONLY, Opus)

Adversarial gate on DIS-21 + the campaign-concluding sweep. Every load-bearing
vendor claim re-derived FIRST-HAND from `vendor/rapidgzip/` (not from result prose),
because the campaign has misread this exact region twice. gzippy-side constants
spot-checked too. HEAD per ledger d56cb0f5 / origin 7bf26096.

---

## ONE-LINE VERDICT

**The vendor re-read in DIS-21 point 1 is CORRECT this time — verified line-by-line.
`m_window16` is a 65536-u16 modulo-AND RING, `resolveBackreference` is the modular
single-memcpy + backward marker scan, and `read()`→`DecodedData::append`→
`appendToEquallySizedChunks` drains into 64Ki-u16 (128 KiB) SEGMENTS. gzippy mirrors
all three byte-for-byte. The "de-frag toward a flat `m_window16`" lever IS a PHANTOM
(point 1 SOUND; this RETROACTIVELY corrects the prior rg-marker gate's "CLAIM B is a
faithful lever" to a misread). The A/B (point 2) is a clean rule-3 low-ceiling TIE
and is SOUND. The attribution reconciliation (point 3) is arithmetically clean and
mechanistically plausible but rests on an UN-ISOLATED inference (rg's output/backref
share was never perf-isolated — it is inlined inside the `Block::read` monolith); it
is licensed by point 1's structural identity, not measured — label it an inference.
The sweeping conclusion (point 4) is SOUND IN DIRECTION but OVER-CLAIMS by (i)
conflating T1 with T4 under "inner-Huffman asm plateau" — at T1 the isal gap is
PROVABLY the chunking-pipeline serialization (DIS-15, single-shot beats rg by 20%),
NOT the Huffman symbol rate (markers≈0 at T1); and (ii) "ONLY irreducible" absolutism
where a bounded-small scheduling/handoff slice was never positively zeroed and the
asm path EXISTS but is user-gated below its 0.85 bar. The diagnostic phase IS
legitimately complete (every named lever refuted with a mechanism, residual
localized); what remains is DECISIONS (asm integration, T1 single-shot routing fork,
the BAR-1 bar itself) + one orthogonal CORRECTNESS coverage gap (OPEN-3), not open
diagnostic levers.**

---

## POINT 1 — "gzippy's u16 output/backref/segment machinery is byte-for-byte faithful
to rapidgzip; the de-frag-to-flat lever is a PHANTOM." → **SOUND (vendor + gzippy
verified first-hand).**

(a) **`deflate.hpp:805` `using PreDecodedBuffer = std::array<uint16_t, 2 *
MAX_WINDOW_SIZE>;`** — VERIFIED. 2 × 32 KiB = 65536 u16. The comment block
`:794-801` is explicit: "circular buffer ... Round up to power of two ... the
circular buffer index modulo operation [can be] a simple bitwise 'and'." `:926`
`alignas(64) PreDecodedBuffer m_window16`. `appendToWindow` (`:1319-1321`):
`window[m_windowPosition] = decodedSymbol; m_windowPosition++; m_windowPosition %=
window.size();` — a MODULAR RING WRITE, not a flat append. **This is a RING, not a
"flat per-chunk linear buffer." The premise the de-frag lever rested on is FALSE.**

(b) **`resolveBackreference` (`:1349-1390`)** — VERIFIED ≡ gzippy `emit_backref_ring`:
`:1369` `offset = (m_windowPosition + window.size() - distance) % window.size();`
(modular offset); `:1374` no-wrap fast path `m_windowPosition + length <
window.size()`; `:1376` `std::memcpy(&window[m_windowPosition], &window[offset],
length * sizeof(window.front()))` (single memcpy, no per-byte modulo); `:1379-1389`
backward marker scan over the just-copied range. Exactly the structure DIS-21 cites.

(c) **`read()` `:1288` `result.dataWithMarkers = lastBuffers(m_window16,
m_windowPosition, nBytesRead)`** (VIEWS into the ring) → **`DecodedData::append` →
`appendToEquallySizedChunks` (`DecodedData.hpp:243-275`)** with `ALLOCATION_CHUNK_SIZE
= 128_Ki` → 64Ki u16 elements per segment (`:247`), copied via `insert` into a
`dataWithMarkers` segment LIST (`:273-274`). VERIFIED. rg HAS the ring, HAS the
per-read view-drain, HAS the segment list.

gzippy side spot-checked: `marker_inflate.rs:232 RING_SIZE = 2 * MAX_WINDOW_SIZE`
(=65536, byte-for-byte (a)); `lut_bulk_inflate.rs:849/924/980` `pos % RING_SIZE` ring
writes; `segmented_markers.rs:86 SEGMENT_ELEMENTS = 64 * 1024` = rg's 128 KiB /
sizeof(u16) (byte-for-byte (c)). **gzippy's `output_ring` + `drain_to_output` +
`SegmentedU16::push_slice` is a faithful structural port of rg's `m_window16` +
`lastBuffers`-drain + `appendToEquallySizedChunks`.**

CONSEQUENCE (must be propagated): the prior `rg-marker-completing-gate-verdict.md`
CLAIM B verdict — "SOUND (faithful/architectural lever to flatten toward
m_window16)" — was itself built on the SAME flat-buffer misread DIS-21 now corrects.
DIS-21 supersedes it: there is nothing flat to converge to; a flat buffer would
DIVERGE (bias-guardrail violation). **Point 1 is the strongest item in the
submission and is correct.** VERDICT: **SOUND.**

---

## POINT 2 — "GZIPPY_FLAT_BACKREF A/B (word-copy→plain memcpy) is a clean T4 TIE
(−1ms median, −96M/1.5% instr, wall flat) = rule-3 low ceiling." → **SOUND.**

I cannot re-run the bench (read-only, no frozen guest), but the result is internally
coherent and is the textbook rule-3 signature: an instruction win (−96M) that does
NOT move the wall (−1ms median, well inside the ±30ms / 5–6% spread). The caveat
DIS-21 states itself is the right one and is load-bearing: this swaps ONLY the copy
MECHANISM (~96M), not the full 0.993e9 `emit_backref_ring` — and per point 1 the
remainder (offset-modulo + marker-scan + the `push_slice` drain) are EXACT vendor
counterparts, hence NOT faithfully removable. So the A/B bounds the only
faithfully-removable sub-piece and finds it wall-slack. Honest framing. VERDICT:
**SOUND** (measurement unverifiable here, but logic and rule-3 classification correct;
the `push_slice` half was not separately measured but is correctly disposed of by
point 1 as an exact `appendToEquallySizedChunks` counterpart).

---

## POINT 3 — the attribution reconciliation (gz inner 3.338e9 = 1.70e9 shared
output/backref + 1.61e9 pure-Rust Huffman; rg inner 2.246e9 = 1.70e9 shared + 0.5e9
ISA-L-Huffman; excess +1.09e9 ≈ 1.61e9 − 0.5e9). → **SOUND ARITHMETIC / FIX-NEEDED
LABEL (it is a structural INFERENCE, not an isolated measurement).**

Arithmetic checks against DIS-19's own per-symbol numbers:
- gz output/backref = emit_backref_ring 15.89% + push_slice 11.29% = 27.18% × 6.252e9
  = 1.699e9 ≈ 1.70e9. ✓
- gz pure-Rust Huffman = read_internal_compressed 19.57 + ShortBitsCached 2.91 +
  unified_marker 2.46 + LutLitLenCode 0.80 + header/init 0.47 = 26.21% = 1.639e9
  ≈ 1.61e9. ✓
- 1.70 + 1.61 = 3.31 ≈ gz inner 3.338e9 ✓; 1.70 + 0.5 = 2.20 ≈ rg inner 2.246e9 ✓;
  excess 1.61 − 0.5 = 1.11 ≈ +1.09e9 ✓.

**THE FIX-NEEDED (exactly the prompt's Q3): rg's output/backref share was NEVER
perf-isolated.** In rg, `appendToWindow` and `resolveBackreference` are INLINED into
the `Block<false>::read` monolith (DIS-19 measured `Block<false>::read` at 41.75% as
ONE symbol). The "rg pays ~1.70e9 output/backref" figure is therefore an ASSUMPTION
that rg's inlined output/backref instructions equal gzippy's 1.70e9 — licensed by
point 1's structural byte-for-byte identity (same ring, same resolveBackref, same
segment drain, same ~73M markers / matched byte fraction), but NOT independently
measured. Two minor mechanistic riders:
  - DIS-21 point 2 itself shows gz's backref copy is ~96M instr HEAVIER than rg's
    (word-copy vs memcpy), so "equal" is approximate and slightly favors gz being
    heavier on output/backref → the true Huffman-symbol-rate residual is marginally
    LESS than +1.09e9.
  - rg's per-symbol marker Huffman primitive is `HuffmanCodingISAL::decode` (a C++
    per-symbol LUT, confirmed by the prior rg-marker gate), inlined in `Block::read`
    — NOT the NASM `isal_inflate` kernel (that is the separate clean-tail flip). The
    "rg ISA-L-Huffman ~0.5e9" is the derived residual (2.246 − 1.70), consistent and
    mechanistically the right object, but also derived-not-measured.

So point 3's SHAPE is right and is now structurally supported, but it should be
stated as an INFERENCE from point 1, not as an isolated apples-to-apples measurement
of rg's two halves. The headline ("the irreducible residual is the marker-prefix
Huffman symbol rate") does NOT depend on point 3 alone — it is triangulated by point
1 (structural), point 2 (the only removable output piece TIEs), and VAR_VIII (0.667x
plateau). Robust to point 3 being inferential. VERDICT: **SOUND (arithmetic +
mechanism) / FIX-NEEDED (relabel as a point-1-licensed inference; rg output/backref
not isolated).**

---

## POINT 4 — THE SWEEPING CONCLUSION. → **SOUND IN DIRECTION / FIX-NEEDED (two
over-claims) — and the parallel-scheduling residual IS folded in, with one
bounded-small loose thread.**

### What is SOUND
- T8 ties (0.985–0.990x, at the BAR-1 threshold; parallelism slack-masks the engine —
  TIE-5). ✓
- isal T4 ≈ 0.90x residual, after the night's refutations, IS dominated by the
  marker-prefix work and is asm-bounded: LEV-1 (even REAL ISA-L = 0.900x at T4,
  removal-oracle, tight); disentangle-placement (granting rg-grade boundaries with
  the engine HELD = TIE-to-WORSE, −34ms → placement is NOT the lever, the lever is
  the gated engine swap on the marker prefix); DIS-21 (output/backref de-frag refuted
  as shared + TIE); DIS-17 (contention/false-sharing/cache all NULL; the gap is +40%
  WORK). The residual lands on the marker-prefix Huffman symbol rate
  (read_internal_compressed vs HuffmanCodingISAL), and VAR_VIII proves that axis
  plateaus at 0.667x ISA-L even with a full register-pinned kernel. ✓
- native carries the above PLUS the 0.667x clean-path engine floor (VAR_VIII), so
  native pure-Rust no-FFI >=0.99-every-T is unreachable on the evidence: native T4
  can't clear 0.99 even with a perfect engine (LEV-1), and its engine isn't perfect;
  native T1 single-shot would still run the 0.667x pure-Rust engine (single-shot only
  beats rg when it IS ISA-L), so it can't recover T1 the way isal can. ✓ (Strongly
  supported; "unreachable" is a leading-and-well-evidenced conclusion, the var8-gate
  itself logged it as "leading hypothesis, not removal-proved" for the pipeline term —
  the wording should retain that one notch of humility.)

### OVER-CLAIM #1 (FIX-NEEDED) — T1/T4 are CONFLATED
The sentence "the ONLY irreducible isal low-T gap is the marker-prefix Huffman SYMBOL
RATE … BAR-1 at low-T is gated by the inner-Huffman asm plateau for BOTH builds" is
FALSE for **isal at T1**. DIS-15 (single-shot removal oracle, removal-grade) proved
isal T1's 0.899x gap is the per-chunk ParallelSM PIPELINE SERIALIZATION (each chunk
waits the prior's 32 KiB window → chunking buys nothing at 1 thread), and that
single-shot ISA-L (no chunking) = **1.197x rg (beats rg by 20%)** with the SAME igzip
kernel. At T1 markers are ≈0 (flip_to_clean=0, windows always present sequentially —
OPEN-1, DIS-15). So the isal T1 gap is NOT the Huffman symbol rate and is RECOVERABLE
via a routing fix. The accurate decomposition:
  - isal **T1**: chunking-pipeline serialization (recoverable via single-shot; beats
    rg). NOT engine, NOT Huffman.
  - isal **T4**: marker-prefix Huffman symbol rate (asm-bounded), placement-entangled.
  - native (all T): + 0.667x clean-path engine floor.
  - T8: ties (slack).
The conclusion must split T1 from T4; collapsing both into "inner-Huffman asm
plateau" mis-states the most-recoverable cell.

### OVER-CLAIM #2 (FIX-NEEDED) — "ONLY irreducible" absolutism
- "irreducible": at isal T4 the residual is asm-BOUNDED, not irreducible — a port
  path EXISTS (VAR_VIII, 0.667x) but is user-gated below its own 0.85 integration
  bar. Precise claim: "the dominant residual is the marker-prefix Huffman symbol rate,
  reducible only via an asm/FFI engine the faithful-non-asm path cannot reach."
- "ONLY": the Red-team ruling (ledger §G) already flagged that the genuinely
  non-engine scheduling/handoff slice within LEV-2's 0.101x bucket is "SMALLER than
  0.101x and currently UNSIZED." disentangle-placement zeroed the PLACEMENT slice and
  FFI is bounded-small (seed-all routes all 17 chunks through ISA-L FFI yet beats rg,
  so FFI/handoff cannot be a large term), but a small scheduling/handoff slice was
  never positively driven to zero. It is bounded-small, NOT lever-authorizing, but
  "ONLY" overstates closure. Say "the dominant residual, with any remaining
  scheduling/handoff slice bounded-small and unsized."

### Is the parallel-scheduling residual (DIS-16/17) folded in or still open?
**Folded in, not separately open.** DIS-16's "deeper parallel-scheduling" phrasing was
SUPERSEDED by DIS-17: contention NULL, false-sharing HITM=noise, locks ~0, cache
EQUAL-OR-BETTER → the T4 gap is +40% INSTRUCTION COUNT (work), located by DIS-18 in
the pure-Rust u16 marker engine. OPEN-2 (decode_NOT_STARTED head-of-line stalls:
worker-saturation vs prefetch-horizon) is effectively resolved toward
WORKER-SATURATION/engine by disentangle-placement (perfect placement on the verified
binary did NOT help → not a horizon/placement gap). The owed "head-of-line /
window-publish serialization removal oracle" was never run AS a named standalone
oracle, but its two candidate mechanisms (contention, placement) were each removed and
fired NULL/TIE, so the residual attributes (not perturbation-isolates) to engine work.
This is the one honest soft spot: the T4 residual's localization to "marker-prefix
Huffman" is an ATTRIBUTION-after-refutation strengthened by disentangle-placement,
not a single clean removal oracle on the scheduling term. It is sound enough to
conclude the diagnostic, but the wording should say "attributed" not "proven
irreducible."

---

## IS THE DIAGNOSTIC PHASE LEGITIMATELY COMPLETE?

**YES for the perf-attribution question, with the point-4 wording corrected.** Every
named alternative lever has been refuted with a MECHANISM (rule 7), not a bare TIE:
placement (DIS-6/disentangle), output shared-floor, marker-bootstrap shared,
contention/false-sharing/cache (DIS-17), consumer-lean (DIS-16), window-propagation
(DIS-18), de-frag flat-buffer (DIS-20 criticality but DIS-21 low-ceiling +
phantom-source), offset-prefetch (DIS-6). The residual is localized: isal-T1 =
recoverable pipeline serialization; isal-T4 = asm-bounded marker-prefix symbol rate;
native = + 0.667x engine floor; T8 = ties. The engine machine-code identity is
disasm-closed; flip-to-clean is confirmed convergent. That is a coherent, heavily
red-teamed terminus.

### What remains GENUINELY OPEN (none is a diagnostic-blocking perf lever)
1. **isal T1 routing fork** — single-shot ISA-L beats rg (DIS-15) but is a path-fork
   off the "ONE PRODUCTION PATH / pure-Rust-sole" goal. A USER GOAL DECISION, not a
   diagnostic gap.
2. **isal T4 asm engine integration** — VAR_VIII 0.667x is below its own 0.85 bar; a
   USER-GATED decision, not a diagnostic gap.
3. **The bounded-small, unsized scheduling/handoff slice** inside T4's residual
   (OPEN-1's last thread). Bounded small (FFI bounded by seed-all-beats-rg), not
   lever-authorizing, but not positively zeroed. The only honest open measurement —
   and it is small.
4. **OPEN-3** stored/fixed SYNC_FLUSH clean-tail coverage — a CORRECTNESS coverage
   gap, gated, ORTHOGONAL to the low-T wall conclusion.
5. **BAR-1 itself** (>=0.99 every T) — a user-set bar the user may revisit given (1)
   the T1 fork is a goal-shape decision and (2) native's engine floor.

---

## PER-CLAIM SCORECARD

| # | claim | verdict |
|---|-------|---------|
| 1 | u16 output/backref/segment machinery is byte-for-byte faithful to rg; de-frag-to-flat is a phantom | **SOUND** — verified first-hand: deflate.hpp:805 ring + :1319-1321 modular write + :1369-1390 modular single-memcpy/marker-scan + :1288→DecodedData.hpp:243-275 64Ki-u16 segment drain; gzippy RING_SIZE=2*MAX_WINDOW_SIZE, SEGMENT_ELEMENTS=64Ki mirror it. Retroactively corrects the prior rg-marker gate's CLAIM-B "faithful lever." |
| 2 | FLAT_BACKREF A/B is a rule-3 low-ceiling TIE | **SOUND** (unverifiable numbers, but coherent rule-3 signature; correct that it bounds only the removable copy-mechanism, rest = exact vendor counterparts) |
| 3 | reconciliation: +1.09e9 excess ≈ gz pure-Rust Huffman − rg ISA-L Huffman | **SOUND arithmetic + mechanism / FIX-NEEDED label** — rg's output/backref was NOT perf-isolated (inlined in Block::read 41.75% monolith); the 1.70e9-for-rg is a point-1-licensed INFERENCE, not a measurement. Headline survives via triangulation. |
| 4 | the ONLY irreducible isal low-T gap is the marker-prefix Huffman symbol rate; BAR-1 low-T gated by asm plateau both builds; native >=0.99-every-T unreachable | **SOUND DIRECTION / FIX-NEEDED** — (i) conflates T1 (provably pipeline-serialization, single-shot beats rg, NOT Huffman) with T4 (Huffman symbol rate); (ii) "ONLY/irreducible" overstates — asm path exists-but-gated, scheduling slice bounded-small-but-unsized. T4 scheduling residual IS folded in (DIS-17 NULL + disentangle-placement), attributed not perturbation-isolated. native-unreachable is well-supported, keep one notch of "leading-hypothesis" humility. |
| Complete? | diagnostic phase complete with this conclusion | **YES** for perf attribution (every lever refuted with a mechanism, residual localized), CONTINGENT on correcting point-4's T1/T4 conflation + "only/irreducible" wording. Remaining items are user DECISIONS (single-shot fork, asm integration, BAR-1 bar) + one orthogonal CORRECTNESS gap (OPEN-3) + one bounded-small unsized scheduling slice — none an open diagnostic perf lever. |

No edits to src/. No orphan processes (read-only audit). Synchronous Opus gate as
DIS-21 requested.
