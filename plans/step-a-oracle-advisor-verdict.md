# STEP-A oracle advisor verdict — adversarial disproof (READ-ONLY)

Independent disproof-driven Opus advisor. HEAD a49c357, branch reimplement-isa-l.
All numbers below were re-derived FIRST-HAND from the cited traces
(`/tmp/gzippy-locked-fulcrum-20260607-000148/artifacts-fulcrum/`) with the leader's
own scripts (`/tmp/p1_makespan.py`) plus an independent nesting analyzer
(`/tmp/nest.py`), not taken from the leader's summary.

---
## TL;DR verdicts

- **ORACLE-P (placement removal): REFUTE the "PLACEMENT SUFFICIENT" verdict; downgrade
  to PLACEMENT NECESSARY-BUT-INSUFFICIENT.** The numbers reproduce, but the *conclusion*
  is built on (1) an apples-to-oranges comparison (gzippy's no-ramp floor vs rapidgzip's
  with-ramp wall) and (2) a mischaracterization of the speculative/marker premium as
  "redundant re-decode." Applied consistently, the leader's OWN arithmetic lands
  placement-perfect gzippy at **~0.56–0.66s vs rapidgzip 0.524s — a ~7–26% LOSS**, which
  is squarely the falsifier's own pre-registered NECESSARY-BUT-INSUFFICIENT band
  (0.55, 0.75].
- **ORACLE-C (clean-compute removal): CORROBORATE-WITH-CAVEATS** the leader's own GREY /
  ~0.4–0.7s read. Region-removed proof genuinely passes (decode 6.33s→0.076s). But the
  oracle is degenerate (free decode also frees windows ⇒ publish-chain collapses) and the
  raw 3.63s carries un-subtracted warm/load — it cannot isolate a class-C ceiling. Do not
  treat 0.4–0.7s as a clean number.
- **REACHABILITY:** placement is the single largest lever and the correct FIRST step, but
  it is **necessary-but-insufficient**, and **class-C/engine is CO-PRIMARY, not
  "bounded-secondary."** gzippy's own *clean* per-chunk decode rate (91 ms) is **2.3×**
  rapidgzip's (39 ms) — an engine gap that survives perfect placement. The charter's
  "class-C bounded-secondary" ranking is not supported by the traces.

---
## What reproduced (the leader's busy numbers are SOUND)

| quantity | leader | re-derived | match |
|---|---|---|---|
| gzippy T8 `decode_chunk` busy | 6.33s | **6.329s** | ✓ |
| gzippy T8 `scan_candidate` busy | 5.90s | **5.895s** | ✓ |
| gzippy T8 `isal_stream_inflate` busy | 0.365s | **0.365s** (n=4) | ✓ |
| gzippy T8 busy LPT makespan | 0.824s | **0.824s** | ✓ |
| rapidgzip T8 `decode_chunk` busy | 2.99s | **2.994s** | ✓ |
| rapidgzip T8 makespan | 0.385s | **0.385s** | ✓ |
| actual/makespan ratio (both tools) | 1.36 | **1.36 / 1.36** | ✓ |

The busy-extraction and makespan computer are valid; the self-tests pass. The leader's
**numbers** are trustworthy. The dispute is entirely with the **inference** drawn from
them.

---
## ORACLE-P — REFUTE the "sufficient" claim

### B (verify first-hand). DONE — see table above. scan_candidate = 5.9s, makespan
method sound, 1.36 on both tools confirmed. T1=3.734s is a REAL measured number
(`plans/orchestrator-status.md:676,690`: "T1 F=0 3.7340s", the slow-injection positive
control at T1, silesia-large 503MB, FORCE_PARALLEL_SM). So B does not refute — it
corroborates the inputs.

### The structural fact the leader's framing hides (nesting analysis)
`/tmp/nest.py` on the gzippy T8 trace gives the span tree:

```
worker.decode_chunk (6329ms incl)
└ worker.decode
  ├ worker.scan_run → worker.scan_candidate (5895ms incl, 1283ms SELF)   n=35
  │   ├ worker.block_header (234ms self, 7780 events)
  │   └ worker.block_body   (4378ms SELF, 7720 events)   ← genuine DEFLATE decode
  └ worker.isal_stream_inflate (365ms self)               n=4  (clean path)
```

`scan_candidate` is NOT "scan overhead." **4378 ms of its 5895 ms is `block_body` —
real first-pass Huffman/DEFLATE decoding of 7720 blocks** (source:
`chunk_fetcher.rs:3255` → `try_speculative_decode_candidate`). Calling the whole 5.9s
"speculative RE-decode" (falsifier line 155: "redundant speculative re-decode") is a
**mischaracterization**: each chunk is decoded ONCE, just via the expensive marker path
because its window is not known ahead of the consumer frontier. "Decoded once, expensively"
≠ "decoded twice." There is no trace evidence of 2.6s of *duplicate* byte decoding.

### C (is 3.734 single-pass, and same engine?) — THE LOAD-BEARING ERROR
Per-chunk rates, re-derived first-hand:
- gzippy **clean** (`isal_stream_inflate`): 365 ms / 4 = **91 ms/chunk**
- gzippy **marker** (`scan_candidate` incl): 5895 ms / 35 = **168 ms/chunk**
- T1 wall 3.734s / ~42 chunks ≈ **89 ms/chunk** ⇒ **T1 is the CLEAN rate.**

So T1=3.734s is gzippy decoding every chunk via the **clean** path (at T1 the single
worker always has the predecessor window, so no speculation). T8's 6.33s is the **marker**
path (168 ms/chunk). The 6.33 − 3.734 = 2.6s gap is the **marker/speculative PREMIUM
(~77 ms/chunk × 35), not redundant re-decode.** The leader's makespan rescales the T8
*shape* down to the 3.734s *clean* total — i.e. it silently assumes **perfect placement
converts every marker decode into a clean decode.** That assumption is the entire source
of the 0.41–0.49 floor, and it is not justified:

1. **Parallel decode inherently needs speculation.** Worker N's clean window is worker
   N−1's *output*; in a parallel pipeline it is not available ahead. rapidgzip pays this
   too — CLAUDE.md: "rapidgzip carries the SAME u16 marker machinery… 31.25%
   replaced-marker symbols." So "all-clean parallel decode" is not a state any faithful
   port reaches; the marker premium is structural, not pure placement slack.
2. **Even gzippy's clean rate loses.** gzippy clean = 91 ms/chunk vs rapidgzip clean =
   39 ms/chunk (1297ms/33). gzippy's clean engine is **2.3× slower**. Best-case
   placement (all chunks clean) ⇒ floor 42×91/8 ≈ 0.48s, ×1.36 ≈ **0.65s > 0.524s.**
   The engine residual survives perfect placement.

### D / the ramp self-refutation — the single cleanest disproof
The "positive control" (actual/makespan = 1.36 on both tools) is real but is **misused to
manufacture the verdict.** The 1.36 is just actual÷makespan (gzippy 1.121/0.824; rapidgzip
0.524/0.385 — near-tautological). If that ramp applies to gzippy's placement-free
operation, it must be applied **consistently**:

| comparison (consistent basis) | gzippy | rapidgzip | gzippy verdict |
|---|---|---|---|
| floor ↔ floor (no ramp) | 0.41–0.49s | **0.385s** | 7–27% HIGHER |
| wall ↔ wall (×1.36 ramp) | **0.56–0.66s** | 0.524s | 7–26% HIGHER |

**Either consistent comparison puts placement-perfect gzippy ABOVE rapidgzip.** The
"sufficient" verdict appears ONLY by comparing gzippy's *no-ramp floor* (0.41–0.49) to
rapidgzip's *with-ramp wall* (0.524) — a floor-vs-wall mismatch. The leader's own
falsifier pre-registered "NECESSARY-BUT-INSUFFICIENT: FLOOR_P in (0.55, 0.75]"; the
ramp-consistent number (0.56–0.66) **lands exactly in that band.** Per CLAUDE.md (Δ <
spread ⇒ TIE), a 24% gap is not a tie. **Oracle-P self-refutes "sufficient."**

### A (is the floor reachable at all?) — secondary, but compounds the over-claim
The memory documents **3 FAILED** consumer-confirmation prefetch attempts (the
~1-chunk confirmation lead is fundamentally too short to prefetch+decode the overshoot
tail in time; all reverted at 43f1685). So even the *placement* portion of the floor is
not demonstrably reachable in a faithful port — the leader's own CAVEAT concedes this
("does NOT prove the redundant re-decode is FULLY eliminable"). A floor that assumes
(a) perfect marker→clean conversion AND (b) perfect load balance AND (c) the
unsolved-3×-over prefetch gap is solved is an **optimistic lower bound stacked on three
unproven removals** — exactly the "slow-down slope ≠ speed-up ceiling" trap rule 3 warns
against, in the speed-up direction.

### Cross-check: spec_pred_off is a no-op (independent corroboration)
`trace_gzippy_spec_pred_off_T8.json` (GZIPPY_SPEC_PRED_CLEAN=0) decode busy = **6.3296s**,
byte-identical to spec-pred-ON 6.3291s. gzippy's clean-decode-ahead prediction currently
converts **~nothing** (consistent with only 4/42 chunks hitting the clean path). This
confirms BOTH (i) the placement headroom is large (almost everything is marker-decoded)
AND (ii) the residual after capturing it is the engine (clean rate 91 ms). It does NOT
support "the 5.9s is removable placement penalty."

---
## F — ADJUDICATION of the fulcrum arbiter ("RATE 15%") vs leader ("placement")

**Neither framing is the verdict; the leader's reframe is HALF-right and over-reaches.**

- The arbiter sees workers BUSY (block_body = genuine first-pass decode), almost no
  consumer idle, and calls it RATE-bound, lever ≈ decode speed ~15%. It is **right that
  the 5.9s is real busy work, not idle stall** — block_body (4378ms) is the engine
  literally decoding bytes.
- The leader is **right that part of that busy is a placement-coupled premium**: the
  marker path (168 ms) vs the clean path (91 ms) is a 77 ms/chunk surcharge that
  window-ahead placement would reduce.
- The leader **over-reaches** by collapsing the two into one ("RATE-bound and
  placement-bound are the same phenomenon") and then using that to dismiss the engine
  residual entirely. The makespan arithmetic refuses this: removing the *entire* marker
  premium (best-case placement) still leaves gzippy at its CLEAN rate of 91 ms/chunk,
  i.e. ~0.65s with ramp — **2.3× slower per clean chunk than rapidgzip's 39 ms.** That
  surviving gap is exactly what the arbiter calls "decode speed," and it is NOT 0%.

**Bottom line on F:** the wall is jointly bounded — placement (marker→clean premium +
4 cold re-decodes, ~1.121 → ~0.65–0.79) AND engine (clean 91 vs 39 ms, floor ~0.6–0.65
even idealized). The arbiter's "PLACEMENT 0.0%" idle-attribution misses the marker
premium (correct critique by the leader); but the arbiter's "decode rate matters" is
**vindicated** by the surviving 2.3× clean-rate gap. The leader's "placement is
sufficient, decode rate is bounded-secondary" is the over-claim. Call it **co-primary.**

---
## ORACLE-C — CORROBORATE-WITH-CAVEATS (the leader's own GREY read is honest)

- **Region-removed proof: genuinely PASSES.** Independent of the leader's note, the
  mechanism is sound: `decode_bypass` replay drops `decode_chunk` busy 6.33s → 0.076s
  (only the 2-miss real-decode fallback). The region is actually removed; this is a
  removal oracle, not attribution.
- **But the magnitude is uninterpretable, for the reasons the leader already flagged
  (credit to the leader for self-catching both):**
  - **Inflation:** the raw CEIL_FLOOR_A = 3.6298s is whole-process wall and still
    contains the ~3.1s warm prebuilt-map rebuild + ~0.4s 656MB capture-load. The
    `warm_prebuilt()` fix moved the rebuild before `drive_t0` but the harness times the
    whole process, so it is NOT subtracted. ⇒ post-subtraction ≈ ~0.5s, but that
    subtraction is itself a hand-correction, not a measured floor.
  - **Deflation (the deeper, structural defect):** decode≈0 ⇒ windows arrive instantly
    ⇒ `L_resolve` collapses (162µs/chunk vs 19.93ms at real decode speed). The publish
    chain that would bind at real decode speed is **artificially free** in this oracle.
    So the floor-trace critpath (336ms) is an under-estimate.
- **Consequence:** Oracle-C is a **degenerate** instrument here — removing clean COMPUTE
  also removes the window-arrival latency that drives the publish chain, so it cannot
  separate "class-C compute ceiling" from "publish-chain ceiling." The 0.4–0.7s bracket
  is real as a *bracket* but must not be read as "free clean compute reaches ~0.5s."
  The owed re-run — **fast-but-REAL decode (CRC-kept RAM replay, persistent process so
  warm is out-of-wall)** — is the correct de-entangler and is still owed. Verdict GREY is
  the right call.

Note the tension Oracle-C creates with placement-primary: its bracket (0.4–0.7) STRADDLES
0.524, which would superficially say "free clean compute might be sufficient" — the exact
opposite of placement-primary. That this oracle and Oracle-P point in *different*
directions is itself the signal that **neither single lever is cleanly sufficient.**

---
## BOTTOM LINE — reachability

**Placement is NECESSARY-BUT-INSUFFICIENT. The "PLACEMENT SUFFICIENT to reach the tie"
verdict is an OVER-CLAIM and should be struck.**

1. By the leader's OWN ramp (the validated positive control), applied consistently,
   placement-perfect gzippy = **0.56–0.66s vs rapidgzip 0.524s** — a 7–26% loss, in the
   pre-registered NECESSARY-BUT-INSUFFICIENT band. "Sufficient" came from a floor-vs-wall
   mismatch.
2. The 5.9s `scan_candidate` is **first-pass marker decode**, not "redundant re-decode";
   the marker→clean premium is partly structural (rapidgzip marker-decodes too) and only
   partly recoverable.
3. A genuine **engine residual survives perfect placement**: gzippy's clean per-chunk
   rate (91 ms) is **2.3× rapidgzip's (39 ms)**. So class-C/engine is **CO-PRIMARY**, not
   "bounded-secondary." The charter's ranking is not supported by these traces.

**Recommendation:** keep placement as the FIRST work item (it is the single largest lever:
~1.12 → ~0.65–0.79s, and it is the faithful-port mandate) — but **strike "sufficient,"**
record the surviving 2.3× clean-engine gap as a co-primary lever, and DO NOT close class-C
as bounded-secondary on the strength of Oracle-C (degenerate) or the pre-gate 11–29%
bracket (which measured the CLEAN-loop slow-injection, a different quantity than the
marker-vs-clean engine-rate gap exposed here). The honest sequence is: port the scheduler
(placement) FIRST, re-measure, and expect a residual ~0.65s that then requires the engine
(class-C / clean-rate) to reach the tie.

### One concrete item the leader should add before STEP-C
The clean-rate gap (gzippy 91 ms vs rapidgzip 39 ms/clean chunk) is the cleanest, least
entangled number in this whole analysis and it is a DIRECT engine signal that escaped
both oracles. It deserves its own removal oracle (a CLEAN-only T8 run — force all chunks
through `isal_stream_inflate` with predecessor windows, measure busy) to set the true
class-C/engine ceiling, since Oracle-C (decode≈0) over-removed and Oracle-P (rescale to
clean) assumed it away.
