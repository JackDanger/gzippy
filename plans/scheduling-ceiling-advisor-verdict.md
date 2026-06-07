# Disproof-advisor verdict — scheduling/serial ceiling (HEAD f1aceee1)

Independent, read-only, source-verified. Job: BREAK the conclusions, not ratify.
Verdicts cite file:line in this tree.

## TL;DR — single most load-bearing correction

**C3's "F2 fired, pure-scheduling lands at 0.159s = loss" is REFUTED.** The
`0.116 + 0.043 ≈ 0.159` figure is — *by the prereg's own words* (prereg lines
31–34) — a **STRICT UPPER BOUND** on the perfect-overlap wall, because the
additive form serializes warm+drain and so **double-counts the in-order
applyWindow/checksum tail that would overlap decode** under true perfect overlap.
An upper bound that happens to be a loss **cannot fire F2.** F2 (≥0.150) requires
a *lower* bound above 0.150; the only lower bound in evidence is the decode-phase
wall ~0.116 ≈ **0.89× rg's wall** — squarely in TIE/WIN (F1) territory. The
decomposition therefore **straddles F1 and F2**; the honest status is **F3
(partial)** — exactly what the prereg pre-registered as most likely (lines
50–51). And the registered instrument that would actually bound it
(`GZIPPY_PERFECT_OVERLAP`, prereg §Oracle) **was never run.** Concluding F2 from
hand arithmetic with no oracle violates Measurement PROCESS Rule 3 (no
extrapolation through an unlocated knee — the knee here is precisely where
serial-tail-overlap crosses decode-phase-wall) and Rules 4/8 (trust the validated
instrument, not the hand sum). **Run the oracle before claiming a ceiling.**

The brief correctly rejects one bad inference (seedfull-ties ⇒ scheduling-is-the-
binder) and then commits the **symmetric** error (arithmetic-upper-bound-loss ⇒
engine-is-the-binder). Both directions are unresolved for the *same* reason: the
pure-scheduling oracle the prereg defined was never executed.

---

## C1 — "engine premium reaches the wall; the prompt's 'engine slack-masked' is WRONG"
**Verdict: UPHELD-WITH-CAVEATS.**

The load-bearing support is NOT the ratios (Theoretical Optimal 1.47×, Real Decode
1.38×) — those are *attribution* metrics, which CLAUDE.md Measurement Rule 1
explicitly bans as verdicts (analyst-biasable, "have repeatedly manufactured
phantom levers"). The genuinely persuasive evidence is the **HEAD→seedfull A/B**,
both measured and sha-exact: wall 0.174→0.128 (Δ≈0.046) while `future::get` is
held ~fixed (0.089→0.083, Δ≈0.006). ~0.040s of wall improvement came from the
**not-`future::get`** axis, and the thing that changed on that axis is the
engine mode (marker→clean). Holding the consumer-wait roughly constant across a
real A/B and watching the wall move with engine mode is legitimate evidence the
engine reaches the wall.

- CAVEAT 1: seedfull changes BOTH axes; it is clean only because `future::get`
  *happened* to land near-equal that it isolates the engine. That is fortuitous,
  not designed — do not generalize it.
- CAVEAT 2: "engine slack-masked" being false at HEAD does not establish the
  engine is the *dominant* remaining term — only that it is *a* term that reaches
  the wall. C1 (engine reaches the wall) ≠ C3/C4 (engine is THE binder).

## C2 — "scheduling is NOT the binder; future::get mostly overlapped"
**Verdict: REFUTED (overstated).**

The brief's own evidence cuts against it:

1. **`future::get` HALVES T8→T16 (0.089→0.046).** A purely-overlapped/slack
   quantity does **not** scale with core count — slack is slack. Responsiveness to
   added parallel resource is the *signature* of a term on or near the critical
   path (more cores ⇒ chunks ready sooner ⇒ less head-of-line wait). The halving
   is evidence *for* criticality, which the brief reads backwards.
2. **It is an in-order head-of-line block by construction.** `recv_post_process_
   blocking` (chunk_fetcher.rs:3047) blocks the consumer on the *head* chunk's
   post-process future; it harvests ready non-head futures + queues prefetch while
   spinning (3072–3084), so it is *partially* overlapped — but the consumer
   literally cannot advance past a not-ready head. That is the definition of a
   live scheduling term, not proven slack.
3. **The 0.083 seedfull floor is CONFIG-SPECIFIC, not a proven floor.** It is the
   residual under *this* prefetch depth / pool size, not an irreducible bound.
   Memory `project_confirmed_offset_prefetch_gap` records a **known, FIXABLE**
   head-of-line stall (4 stalls at confirmed offsets, ~40% of the T8 wall,
   "fixable, NOT architectural"). That directly contradicts "scheduling is not the
   binder." The registered `GZIPPY_PERFECT_OVERLAP` oracle — which actually
   removes the head-of-line wait rather than inferring its floor from seedfull —
   was never run.

What IS true: `future::get` is engine-*independent* (0.089≈0.083) and overlaps in
the tying cell. But engine-independent ≠ overlapped-away ≠ off-the-critical-path.
The brief conflates the three.

## C3 — "ceiling: pure-scheduling fix → 0.116+0.043≈0.159s = loss (F2 fired)"
**Verdict: REFUTED.** (See TL;DR.) Concretely:

- The additive sum is the prereg's **declared STRICT UPPER BOUND** (prereg
  31–34). Reading an upper bound as "the landing point" is the inversion the
  prereg warned against.
- The additive form **double-counts the serial tail.** Under true perfect overlap
  the in-order applyWindow+checksum of chunk *i* runs concurrently with decode of
  chunk *i+1*; it adds to the wall only by `max(decode_phase_wall,
  in_order_chain)`, not by sum. Since the in-order chain (≈0.043) is *shorter*
  than the decode-phase wall (≈0.116), most of it hides — floor ≈ max(0.116,
  0.043+tail-latency) ≈ 0.12–0.13, **a TIE**, not 0.159.
- Using **rg's** 0.043 serial tail (not gzippy's measured 0.058) makes the sum
  *more* favorable yet still calls it a loss — a tell that the construction is
  being pushed to a predetermined F2.
- Lower bound 0.116 (0.89× rg) and upper bound 0.159 (1.22× rg) **straddle**
  F1/F2 → honest verdict F3, as pre-registered. No oracle was run. This is a
  Rule-3 extrapolation through an unlocated knee.

The one structurally-correct kernel inside C3: the window/clean **coupling is real
in source** — `decode_chunk_with_rapidgzip_impl` branches on window width,
`if initial_window.len() == MAX_WINDOW_SIZE` → seeded CLEAN path (gzip_chunk.rs:790)
vs else → `decode_chunk_unified_marker` marker bootstrap (gzip_chunk.rs:826). So
**seedfull over-removes** (all windows present ⇒ all clean ⇒ engine premium gone),
and seedfull's tie genuinely does **not** prove a pure-scheduling tie. That
refutation of the prompt's premise is sound. But it does not license the inverse
claim; it only says *neither* axis is isolated by the oracles actually run. The
correct status of the pure-scheduling ceiling is **UNBOUNDED by the evidence
presented** — run `GZIPPY_PERFECT_OVERLAP`.

## C4 — "next binder = O(length) backward marker scan in emit_backref_ring::<true>"
**Verdict: UPHELD-WITH-CAVEATS → effectively UNCONFIRMED.**

The scan exists and is real marker-only cost: marker_inflate.rs:3006–3027,
gated `if CONTAINS_MARKERS` (2990), O(length) backward over the just-written
slots. But the source argues *against* it being the dominant ~1.6× term:

- It is **fast-path SKIPPED** whenever `*distance_marker >= distance`
  (3002–3005); the code's own comment (2996–2999) says this "holds for the common
  case once a chunk has decoded a window's worth of clean output."
- On the measured (isal) build the chunk **FLIPS to clean u8 at 32 KiB**
  (`MarkerStep::FlipToClean` → `finish_decode_chunk_with_inexact_offset`,
  gzip_chunk.rs:949–973), **exiting the marker path entirely.** So the scan is
  confined to the per-chunk bootstrap (~first 32 KiB of a multi-MB chunk) — a
  small fraction of decoded output. It is implausible as the prime driver of a
  decodeBlock-SUM gap spread across the whole chunk.
- More plausible remaining-term candidates the brief does not weigh: u16-width
  bulk memory traffic during bootstrap, and the **clean-path** (post-flip u8)
  engine rate vs rg's clean rate — which would persist even after the scan is
  zeroed.

C4 is explicitly a hypothesis ("per the prior advisor", "Next: profile/attack").
Under Rule 1 it must be confirmed by a **causal perturbation** (slow-inject the
scan alone, or oracle it out) before any work-stretch. Do not enter the loop on
attribution.

---

## What to do next (instrument, not arithmetic)
1. **Run the registered `GZIPPY_PERFECT_OVERLAP` oracle** (prereg §Oracle). It is
   the only thing that bounds the pure-scheduling ceiling; its absence is why C2/C3
   are unresolved. Its wall is a *strict upper bound* on perfect overlap — if even
   it ties rg (≤0.137), scheduling reaches the tie and the engine is not the sole
   binder.
2. **Validate it before trusting it** (Rule 4) — two instruments this campaign
   were silently broken; a byte-transparent self-test (output sha unchanged) is
   mandatory.
3. Only after (1): if the oracle lands in F2, *then* perturb the engine, and
   perturb the **backward scan specifically** to confirm/refute C4 rather than
   inferring it from decodeBlock SUM.
