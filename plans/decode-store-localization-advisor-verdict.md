# DISPROOF ADVISOR VERDICT — decode-compute vs store-bandwidth localization

**Headline verdict: UPHELD-WITH-CAVEATS for "the contig clean loop is on-path and both
injection points respond." REFUTED for "decode-compute is the more-robust / localized
binder, distinct from store."** The instrument as built cannot separate a microarchitectural
resource (ALU/table-lookup compute vs store/copy bandwidth) inside a single serial dependent
loop. The near-equal `dec100 ≈ store100` is the *expected* output of the design, not evidence
about which sub-resource binds.

---

## The governing disproof (applies to Q1, Q2, Q4)

`decode_clean_into_contig` is a **single-threaded, serially-dependent** loop: `litlen.decode()`
produces the symbol that the very next store/copy consumes. `GZIPPY_SLOW_DECODE` inserts delay
*after decode()*; `GZIPPY_SLOW_STORE` inserts delay *after the store*. These are two adjacent
points in **the same serial instruction stream**. Inserting X ns of delay at either point
lengthens the *same* dependency chain by ~X ns regardless of whether the true hardware
bottleneck is the load/PEXT ALU path or the store port / cache bandwidth.

Therefore:
- A monotone, non-collapsing response from **both** knobs is guaranteed by construction once the
  loop is on-path — it re-proves what `GZIPPY_SLOW_MODE` already proved (loop on critical path),
  and adds **no** new localization power.
- `dec100 ≈ store100` is the **predicted** result of injecting comparable serial delay at two
  points in one chain. It is not "I can't quite separate them"; it is "this instrument has no
  separating power, full stop." Reading the small residual `dec100 < store100` gap as "decode
  binds more" is over-interpretation of noise within that structural tie.

Slow-injection localizes *regions of a pipeline* (where coz/SLOW_BOOTSTRAP shines — distinct
spans that can independently take more wall). It does **not** localize *microarchitectural
resources within one serial scalar loop*. That requires a different probe: perf counters
(port pressure, `mem_load_retired`, cache-miss stalls), a BMI2-on/off A/B, or an oracle that
removes decode-compute while preserving the stores.

---

## Q1 — "both on-path, decode-compute the more robust binder"?

**VERDICT: split. "Both on-path" UPHELD-WITH-CAVEATS. "decode-compute the more robust binder"
REFUTED.**

- "Both on-path": the monotone off > 50% > 100% with non-collapsing sleep is consistent with
  on-path, but per the argument above this is nearly tautological for any delay inserted into
  the confirmed-on-path serial loop. It does not establish two *separable* binders — it
  establishes one loop that responds to delay inserted anywhere in it.
- "decode-compute more robust": NOT established. The discriminator rests on (a) a sub-noise
  `dec100 < store100` gap and (b) the sleep-vs-spin reading dismantled in Q2. With those
  removed there is no evidence ranking decode above store. Call it **"the clean loop binds; the
  binding sub-resource is unresolved by this instrument."**

---

## Q2 — is the sleep-vs-spin discriminator a valid rule-2 reading, or confounded?

**VERDICT: REFUTED — confounded, and partly *backwards*.**

Two problems:

1. **Sleep OVERSHOOT is the wrong sign for "stronger criticality."** Rule 2 expects the sleep
   control to roughly *preserve* the spin delta (sleep ≈ spin, or slightly *smaller* because the
   spin was turbo-inflated). The brief reports decode **sleep EXCEEDS spin** (0.760 < 0.787,
   0.743 < 0.768, 0.756 < 0.815 — sleep is the *slower* wall). A sleep adding *more* wall than
   the equivalent busy work is the signature of **nanosleep granularity / scheduler wake
   latency overshoot** (Linux `nanosleep` floors at tens of µs; a per-event ns debt rounds up
   massively), not of cleaner criticality. The brief reads its own anomaly as confirmation when
   it is more naturally an instrument artifact.

2. **The event-count asymmetry the brief raises in Q2 is real and decisive.** Store events
   (#literals + #matches) outnumber decode events (#symbols + #matches), so per-event sleep
   debt discharges at a different cadence and hits granularity rounding differently. "decode
   sleep grows, store sleep stays flat" is fully explained by *more, smaller store sleeps
   rounding/batching differently from fewer, larger decode sleeps* — a cadence artifact with
   zero bearing on which resource binds. The discriminator cannot be cleanly attributed to
   criticality.

So the rule-2 comparison here is not clean: sleep and spin differ for reasons (granularity,
event count) orthogonal to turbo, which is the only thing rule 2 is licensed to test.

---

## Q3 — proceed to a decode-side technique now, or is a tighter oracle owed first?

**VERDICT: UPHELD-WITH-CAVEATS — separate the *decision* from the *justification*.**

- **Fine to TRY** BMI2 PEXT/BZHI, packed-u32 multi-symbol LUT, wider refill: they are
  byte-exact, you'd want them regardless, and a byte-exact technique that ties is KEPT
  (rule 7). The verdict on each is its **remove-and-measure wall response** — implementing a
  faster decode *is itself* the compute-removed measurement. That is legitimate per rule 3.
- **NOT licensed** to bank "decode-compute is the located binder" as the reason. That claim is
  unproven (Q1/Q2). Justify the work as "byte-exact engine speed-up, wall-validated," not as
  "localized lever."
- **The ceiling caveat:** `ocl_cf 0.925×` removes the **whole engine** — it bounds
  compute+store **together**, not decode-compute specifically. So 0.79→0.925 (~0.135×) is the
  ceiling for *all* clean-loop decode-side work, and a pure-compute technique can claim at most
  a fraction of it. If you want a bound *specific to decode-compute*, the owed oracle is a
  **decode-compute-removed clean-loop oracle** (e.g. feed a precomputed symbol/length stream so
  decode() is free but stores/copies still run) — that, minus off, isolates compute's share.
  That oracle is owed before any *localization* claim, but not before a byte-exact attempt whose
  own wall delta is the verdict.

---

## Q4 — strongest disproof of "decode-compute is the binder"

**The serial-dependency-chain non-separability argument (top of this doc).** Concretely: build
a **decode-compute-removed oracle** — keep every literal-store and `emit_backref_contig`, but
replace `lut_litlen.decode()` / `dist_hc.decode()` with a pre-decoded symbol/length/dist stream
read from a side array (byte-exact output). Measure the T8 wall.
- If wall ≈ off → decode-compute is **slack**, store/copy bandwidth binds, and the brief's
  conclusion is refuted outright.
- If wall drops toward `ocl_cf` → decode-compute genuinely binds.

That single oracle settles it; the two slow knobs cannot, because they only re-confirm the
loop-as-a-whole is on-path. Secondary disproof: a **BMI2-on vs BMI2-off A/B** on the existing
loop — if PEXT bit-extraction is the binder, toggling it must move the wall; a flat response
falsifies "compute binds" directly and cheaply.

---

## Bottom line for the campaign log

- KEEP: "contig clean FOLD loop is on the T8 critical path" (already established; these knobs
  re-confirm it).
- DROP / DO NOT BANK: "decode-compute is the more-robust binder vs store." Sub-noise gap +
  confounded sleep discriminator + structurally non-separating instrument. Δ between dec and
  store is within-mechanism TIE.
- OWED before any *localization* claim: a decode-compute-removed clean-loop oracle (and/or
  BMI2 on/off A/B). NOT owed before a byte-exact technique attempt that is itself
  wall-validated by remove-and-measure.
