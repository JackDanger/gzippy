# DISPROOF VERDICT — is the T8 binder the u16 marker path, or scheduling/serial?

**Advisor:** independent disproof pass. Read the brief in full and source-verified
every cited `file:line` first-hand (`gzip_chunk.rs`, `marker_inflate.rs`,
`slow_knob.rs`). I did NOT modify any source.

## Verdicts at a glance

| # | Conclusion | Verdict |
|---|-----------|---------|
| 1 | u16 marker path is a MINORITY of the T8 wall (~3.5%), the charter's "58.6% u16 = biggest prize" is FALSIFIED | **UPHELD** |
| 2 | decode-compute ≈30% / scheduling-serial-overlap ≈70% of the T8 wall | **UPHELD-WITH-CAVEATS** (the 70% is "everything that isn't *body*-decode," NOT purely scheduling — the residual absorbs header/table-build compute, which neither knob perturbs) |
| 3 | NEXT = attack the scheduling/serial term, not the u16 path | **UPHELD as direction, with a precondition**: decompose the residual (header/table-build vs decode-WAIT vs serial-WORK) BEFORE committing — do not assume residual == scheduling |

---

## Source verification (what I confirmed first-hand)

1. **The counter is provably mis-named — stronger than the brief states.**
   `BOOTSTRAP_POST_FLIP_U16_BYTES` (`gzip_chunk.rs:97` decl, `:1302` inc) increments
   `(sink_len - before_len)` exactly when `flipped_clean = !block.contains_marker_bytes()`
   is true (`gzip_chunk.rs:1300-1306`). It counts bytes in blocks that ended
   **marker-FREE**. The doc comment (`:91-96`) calls these "bytes decoded into the u16
   marker ring AFTER the flip" — that is the OPPOSITE of what the increment computes,
   and is stale pre-`fc1c965b`. So the old charter reading "58.6% of body bytes take
   the slow u16 path" is not merely imprecise — it inverts the counter's meaning. The
   name says "u16"; the predicate selects marker-free blocks. **Premise correction:
   confirmed, and the charter claim is dead on contact with the source.**

   Note one nuance the brief itself slightly overstates: it equates this 57.5% with
   "bytes ALREADY on the fast u8 path." Not exactly — the per-block predicate
   `!contains_marker_bytes()` (`:1300`) is independent of the chunk-level routing gate
   `ctx.flipped` (`gzip_chunk.rs:1205`, which requires `clean_appended_len() >=
   MAX_WINDOW_SIZE`). A block can be marker-free yet still be decoded in `<true>` u16
   mode if the chunk has not yet accumulated 32 KiB of clean output. So the counter is
   "bytes in marker-free blocks," an imperfect proxy for "u8-path bytes." This does not
   rescue the charter (it still isn't counting slow-u16 bytes), but it means the
   "genuine u16 ≈ 42.5% (inverse)" figure is a block-granularity estimate, not a
   byte-exact one. **Harmless to conclusion #1, which rests on the perturbation, not
   the counter.**

2. **The marker knob is real, isolated to `<true>`, and fires on the live path.**
   - `marker_inflate.rs:1453-1457`: `slow_spin = if CONTAINS_MARKERS { marker_spin_iters() }
     else { spin_iters() }` — const-folds so the marker knob is read ONLY on the `<true>`
     instantiation and the clean knob ONLY on `<false>`. No cross-leak. Confirmed.
   - `marker_inflate.rs:1484`: the VAR_V fast loop is gated `!CONTAINS_MARKERS && ...`, so
     the `<true>` path NEVER takes the fast loop and always reaches the careful-loop inject
     at `:1644`. The marker knob therefore fires on every `<true>` decode event. Confirmed.
   - Liveness: the brief reports MARKER spin moved +21% at T8, which is itself proof the
     site is not dead-code. Accepted.

3. **Calibration provenance (a real weakness, see disproof D1).** `BASE_SPIN = 22`
   (`slow_knob.rs:82`) is calibrated so `F=1.0` "roughly DOUBLES the single-thread
   **clean-loop** wall on arm64 native" (`:43`, `:79-82`). `marker_spin_iters()`
   (`:148-154`) reuses the SAME `BASE_SPIN`. `NS_PER_SPIN_ITER = 0.32` (`:190`) was
   measured on "Rosetta x86_64 native, T1" (`:186-189`). The brief's measurement host
   is the 16c homelab guest. So the per-event injection magnitude is calibrated against
   the CLEAN loop, on different hardware than the run.

---

## Disproof attempts

### D1 — "+200% on the marker knob is not actually +200% of the marker region's time" (calibration confound)
**Strongest attack on conclusion #1's NUMBER.** PROCESS rule #1 requires changing the
region's time *by a known factor*. But `marker_spin_iters` injects `BASE_SPIN*F` per
event where `BASE_SPIN` was tuned to the *clean* loop's per-event compute
(`slow_knob.rs:43,82,148-154`). The `<true>` careful loop does strictly MORE per event
than the clean fast loop — it pays the marker backward-scan, `distance_marker`
bookkeeping (`marker_inflate.rs:1703-1705`), u16 stores, and u16 back-ref emit — so a
fixed `BASE_SPIN*2` is a SMALLER fraction of the marker region's real time than of the
clean region's. The marker "+200%" is probably an *under*-injection ⇒ the +7% sleep
response *under-states* marker criticality.

**Does it break conclusion #1?** No — only its precision. To overturn the QUALITATIVE
claim (marker is a minority, not the >50% "biggest prize"), the calibration would have
to be wrong by >7× (turning a measured +7% into >50%). A clean-vs-marker per-event cost
ratio of 7× is implausible — both are LUT-driven Huffman decodes over the same symbol
stream. Even granting a generous 2× under-injection, marker lands ~7% of wall, still a
minority. **Conclusion #1 survives; the "≈3.5%" point estimate should be reported as a
soft lower-ish bound, with a plausible range of ~3.5–14%.**

### D2 — the sleep control silently regresses the CLEAN path (confound the brief missed)
`slow_yield = yield_kind()` is read UNCONDITIONALLY on both instantiations
(`marker_inflate.rs:1458`, not `CONTAINS_MARKERS`-gated), and the fast-loop gate
(`:1484`) is `!CONTAINS_MARKERS && slow_spin == 0 && !slow_yield`. `GZIPPY_SLOW_KIND` is
global. **Therefore, during a MARKER+SLEEP run, the `<false>` clean instantiation sees
`slow_yield = true` and is knocked off the VAR_V fast loop onto the careful loop —
even though the clean *injection* is zero (`inject` early-returns at `slow_knob.rs:211`).**

Consequence: the MARKER+SLEEP arm's wall = (marker sleep effect) + (clean fast→careful
regression). The +7% reading is therefore an *over*-estimate of marker-only criticality;
the true marker-sleep effect is **< 7%**. This cuts in FAVOR of conclusion #1 (marker
even smaller) but it also means the spin-vs-sleep pair (21% vs 7%) is NOT apples-to-apples:
the spin arm keeps the clean fast loop (for MARKER+SPIN, `GZIPPY_SLOW_KIND` is unset ⇒
`slow_yield=false` ⇒ clean fast loop runs), the sleep arm does not. So the 21→7 collapse
is "turbo artifact + clean-path-regression-in-the-sleep-arm," not pure turbo. The "+7% is
the clean freq-neutral truth" framing is therefore imprecise — but every correction makes
marker SMALLER, so **conclusion #1 is reinforced, not threatened.** I flag it because the
brief presents +7% as a clean reading; it is actually a confounded upper bound.

### D3 — event-coverage vs byte-coverage bias (brief's own angle 3)
The inject fires once per careful-loop outer iteration = once per Huffman codeword EVENT
(`marker_inflate.rs:1641-1644`), but a back-ref event emits 3–258 bytes. The u16 pre-flip
prefix is exactly the back-ref-into-unknown-window region, so it is relatively
back-ref-heavy ⇒ fewer inject events per byte ⇒ the per-byte marker cost is UNDER-sampled.
Like D1, this biases the reading DOWN, so it cannot rescue the charter's 58.6%; it only
widens the upper end of the plausible range. Accepted as a caveat, not a refutation.

### D4 — "the 70% residual is scheduling/serial" is an elimination-by-residual trap (the real hit, on conclusion #2)
**Neither knob perturbs the block header / Huffman table build.** `read_header`
(`gzip_chunk.rs:1222-1232`, timed into `BOOTSTRAP_HEADER_US/CALLS`) runs the precode +
lit/len/dist table construction; both slow knobs live strictly inside the body loop
(`marker_inflate.rs:1644`, `:1484`). The instrumentation block at `gzip_chunk.rs:81-90`
exists precisely because "8.3B flame samples in marker bootstrap need to split" header
from body. So header/table-build compute is DECODE-compute that the perturbation does not
touch and that therefore falls into the "residual." Labeling the whole ~70% residual
"scheduling/serial/overlap" (conclusion #2) absorbs header/table-build (and any T8
memory-bandwidth / cache-contention term the single-thread-calibrated knob can't see)
into "scheduling." **That is the elimination-by-residual trap the brief's own angle 5
warns about — and here it actually bites.** Verdict: the decode-*body*-compute share is
≤30%; the residual is "non-body-decode," which is scheduling/serial PLUS header/table
build PLUS bandwidth — not purely scheduling. **Conclusion #2 → UPHELD-WITH-CAVEATS.**

### D5 — is the spin-vs-sleep turbo story even valid, or is sleep under-injecting? (brief's angle 2)
The sleep control deschedules the worker (`slow_knob.rs:214-232`), which can let the
in-order consumer/other workers catch up, masking real criticality — so sleep could
under-state and the truth sits between 7% and 21%. Combined with D2 (the sleep arm is
*additionally* inflated by the clean-path regression), the two effects push in OPPOSITE
directions, so the net bias on the +7% is genuinely ambiguous. The honest statement is:
marker criticality is bounded above by the spin reading (~21%, turbo-inflated) and is
most plausibly in the high-single-digits-to-low-teens. **Still a minority. Conclusion #1
holds; the "≈3.5%" specific figure does NOT survive as a point estimate — report a range.**

### D6 — could a faster u16 path unbind a scheduling cascade? (Rule-3, brief's angle 4)
A slow-down SLOPE never bounds a speed-up CEILING (CLAUDE.md PROCESS rule #3). A faster
u16 prefix decode could let speculative chunks reach the 32 KiB clean threshold
(`gzip_chunk.rs:1205`) sooner, flip to the u8 path earlier, and let confirmed-offset
chunks resolve before the consumer head-of-line stalls
([[project_confirmed_offset_prefetch_gap]]). The brief correctly preserves this caveat
for the CLEAN kernel but applies it only weakly to the marker path. I extend it: the low
marker SLOW-slope does NOT prove a low marker SPEED-up ceiling. **However** — this is an
argument for *not over-claiming* that the u16 path is worthless, not an argument that it
is the dominant binder. It does not resurrect the 58.6% charter claim. It is a reason to
keep the u16 prefix on the table as a possible cascade-unbind lever, to be settled only
by an ORACLE removal (decode the prefix for free), never by the slope. Conclusion #1's
"not the dominant binder" stands; conclusion #3's "don't work the u16 path" should be
softened to "don't work it BLINDLY — bound its ceiling with a prefix-removal oracle
before fully abandoning it."

---

## Bottom line

- **Conclusion #1 — UPHELD.** The counter is demonstrably mis-named (it selects
  marker-FREE blocks, `gzip_chunk.rs:1300-1306`), so the charter's "58.6% take the slow
  u16 path" is falsified at the source level independent of any measurement. The causal
  perturbation independently confirms the u16 marker path is a minority of the T8 wall.
  Every disproof I mounted on the NUMBER (D1 calibration, D2 sleep-confound, D3 event
  coverage) pushes the figure DOWN or leaves direction intact; the only upward pressure
  (D5 sleep under-injection, D6 cascade-unbind) cannot plausibly cross from minority to
  >50%. **Report the magnitude as a range (~3.5–14%, most-likely high-single-digits),
  not the point estimate 3.5%.**

- **Conclusion #2 — UPHELD-WITH-CAVEATS.** decode-*body*-compute ≤~30% is well supported.
  "≈70% is scheduling/serial/overlap" is NOT — it is an elimination-by-residual that
  silently includes Huffman header/table-build compute (unperturbed by either knob,
  `gzip_chunk.rs:1222-1232`) and any T8 bandwidth/contention term. The residual is
  "non-body," not "scheduling."

- **Conclusion #3 — UPHELD as direction, with a precondition.** Attacking the
  scheduling/serial term is the right next move, but FIRST split the residual: time
  header/table-build (the `BOOTSTRAP_HEADER_*` counters already exist) and run the
  decode-WAIT vs serial-WORK decomposition the brief proposes, so the next work-stretch
  targets a *located* term and not a residual label. And keep a u16-prefix-removal oracle
  in reserve to bound (per Rule-3) whether a faster prefix unbinds a flip/consumer
  cascade before declaring the u16 path fully closed.

**No source files were modified.**
