# B-vs-b reconcile gate — DIS-25 (B work-distribution) vs DIS-26 (b decode-wait)

Independent disproof advisor, read-only, Opus. Source/ledger verified first-hand
(DIS-18/amdahl-verdict-gate, DIS-24 high-t-gate, DIS-25, DIS-26, DIS-5/skipwritev
oracle, var8-gate). No edits, no orphans. Owes the supervisor Opus gate; this is a
self-disproof reconciliation, not a perf measurement.

## ONE-LINE VERDICT

Neither pure (B) nor pure (b). The contradiction is REAL but NARROW, and DIS-26
OVER-REACHES on one label. Resolved decomposition: **engine-W is the proven root in
the LOW-MID-T box (T3–T8) — there worker busy-fraction was measured 91% (DIS-18),
starvation refuted, S already < rg. At HIGH-T-WITH-E-CORES (T16-Ephys/T24) the
engine-vs-under-feeding question is NOT resolved: DIS-26 measured the CONSUMER's
decode-wait but never measured E-CORE WORKER idle, so it cannot label the high-T
DECODE-WAIT bucket "the engine." DIS-25's 2x-E-core-extraction is a parallel-only
effect NOT explained by the (flat) single-core rate, and survives DIS-26.** The
lean-consumer lever IS correctly bounded (≤23ms). Output is a shared floor. The
campaign has converged on the engine-W DIAGNOSIS at low-mid-T but has NOT converged
to "only user decisions remain" — one cheap discriminator (E-core worker
busy-fraction at T16-Ephys) and one owed low-T per-chunk/pipeline removal oracle are
still un-run, and the per-chunk/ParallelSM term is a candidate faithful lever
explicitly not-yet-proved-irreducible.

## Per-claim

### Q1 — DECODE-WAIT = engine, or under-feeding? → **FIX-NEEDED (DIS-26 over-reaches; the contradiction is unresolved in a narrow regime)**

- DECODE-WAIT as measured (consumer blocked on `block_fetcher_get` + `rx_recv_block`)
  is an AGGREGATE that conflates THREE mechanisms: (i) workers busy on gz's heavier W
  (engine), (ii) workers IDLE waiting for dispatch (scheduling/feeding), (iii)
  per-chunk ParallelSM/FFI/chunk-0 pipeline overhead on the worker. DIS-26's decompose
  isolated only that the CONSUMER's own serial publish/post-proc is NOT the under-feeder
  (2–7%, apply_window already on the pool — both SOUND). It did NOT measure WORKER-POOL
  idle, so it cannot attribute DECODE-WAIT to (i) vs (ii)/(iii). Labeling the whole
  bucket "the engine — user-gated asm lever" is an inference, not a measurement.
- The engine-root claim IS proven, but only in a DIFFERENT box: DIS-18/amdahl-gate
  measured gz worker busy-fraction = **91% at T4** (workers MORE utilized than rg) ⇒
  starvation (H2/H3) refuted ⇒ at T3–T8 DECODE-WAIT genuinely = engine-W (+35% W,
  S already < rg). SOUND, but the gate's own scope (CLAIM-2, line 137) is
  "silesia × T3–T8 × fixed-17-chunk." It does NOT cover T16-Ephys + E-cores.
- DIS-25's under-feeding evidence lives precisely in the un-covered regime and is
  NOT a per-core-rate artifact: per-core gz/rg ratio is FLAT (0.898 P / 0.879 E), so
  the E-core engine deficit is uniform and single-core. The 8 added E-cores drop gz
  wall 35ms vs rg 67ms. The per-core rate gap predicts only rg-E-roofline/gz-E-roofline
  = 1180/1038 ≈ **1.14x** of that delta; the measured delta is **1.91x**. So ~3/4 of the
  E-core extraction differential is a PARALLEL-ONLY pipeline/feeding/contention property
  NOT present in the single-core rate — exactly what the consumer-side decompose cannot
  see. DIS-25's (B) is therefore relabeled-not-refuted by DIS-26.
- Can both be true (engine-W root, (B) the symptom, no separate lever)? Only under the
  engine-W-UNDER-BANDWIDTH-CONTENTION reading (heavier W ⇒ more memory traffic ⇒ worse
  parallel scaling). That is plausible and would still be the asm/clean-rate lever — but
  it is INDISTINGUISHABLE at the consumer from genuine E-core under-feeding without the
  worker-idle measurement. **So the reconciliation is not established; a residual
  feeding/scheduling component is un-refuted in the T16-Ephys regime.**

### Q2 — lean-consumer bounded ≤23ms; refutes "publish-off-consumer is THE lever"; consistent with DIS-16 → **SOUND**

- The ceiling is correctly set by Part-1's serial-work bucket (2–7% = ~5/10/23ms at
  T8/T16/T24, growing with chunk count), NOT by the ~1:1 injection slope. DIS-26
  correctly identifies the slope as the consumer-IS-the-wall tautology (inject anywhere
  on an in-order consumer ⇒ wall grows; the ~20% super-linear part is the only true
  downstream-gating signal, and it is small). Using the decompose bucket as the ceiling
  is the right call per rule 3 (slow-down slope ≠ speed-up ceiling).
- This BOUNDS and thereby REFUTES DIS-25's "move publish off the consumer is THE lever"
  (high-t-B-gate already flagged that surface as DIS-6 item (i), not a new lever; DIS-26
  now quantifies it as ≤23ms, sub-dominant, shrinking to ~0 at the T7–8 win window).
- Consistent with DIS-16's T4 lean-consumer TIE (serial work µs-scale at low-T; grows
  only modestly at high-T). SOUND.

### Q3 — serial writev OUTPUT (30–43%): lever or shared floor? → **SOUND (shared floor, not a lever)**

- DIS-5 refuted output-overlap as the path (non-faithful + sub-parity; the 0.88x
  ceiling was a construction). The skipwritev oracle is decisive: with output REMOVED
  and engine matched and NO marker bootstrap, gz still loses 124ms / 0.867x at T1 ⇒
  output is NOT the gzippy-excess. rg pays comparable writev (~15.7ms reconciliation /
  the shared serial drain). The 30–43% is a SHARED serial floor both tools pay, correctly
  not opened as a lever.

### Q4 — convergence → **FIX-NEEDED (converged on the low-mid-T diagnosis; NOT on "only user decisions remain")**

What IS converged (bankable):
- engine-W is the dominant SINGLE-CORE deficit (uniform 0.90x per-core, +35% W,
  clean-rate 2.3x / LEV-4); at T3–T8 it is the proven root (91% busy, starvation
  refuted). User-gated asm, failed its 0.85 bar.
- lean-consumer machinery: real, faithful, BOUNDED ≤23ms (DIS-26).
- output writev: shared floor, refuted-as-path (DIS-5).
- prefetch-DEPTH and out-of-order-publish (DIS-25's two named fix surfaces): dead /
  unfaithful (high-t-B-gate Q3 — vendor-byte-identical depth; rg consumer in-order).

What is NOT converged (two un-refuted faithful items, same lever family):
1. **High-T DECODE-WAIT discrimination (engine-under-contention vs E-core
   under-feeding) is un-run.** The 91%-busy result does NOT extend to the T16-Ephys
   E-core mask; DIS-25's 2x extraction (parallel-only, not single-core rate) survives.
2. **The per-chunk / chunk-0 / ParallelSM-pipeline term** is gzippy-specific 13% /
   124ms at T1 with engine matched AND output removed AND no marker bootstrap. The
   var8-gate explicitly states it "OWES its own removal oracle, NOT removal-proved as
   irreducible," with rg's leaner per-chunk window-map/consumer as the existence proof
   (rule-7 faithful candidate). This per-chunk overhead scales with chunk count (≈3T)
   and is the plausible same mechanism that surfaces as the high-T loss (more chunks ⇒
   more handoffs) and as part of DIS-26's DECODE-WAIT bucket — i.e. DIS-26's "DECODE-WAIT
   = the engine" mislabels this faithful pipeline term as the asm engine.

So the terminal conclusion is NOT yet "the gap is engine-W; remaining moves are USER
decisions + squishy." It is: **engine-W is the proven low-mid-T root and the dominant
single-core deficit (user-gated); a gzippy-specific per-chunk/ParallelSM-pipeline
excess (124ms@T1, faithful candidate) and the high-T E-core feeding question remain
un-refuted and owe one cheap measurement each before the user asm decision is the only
move left.** The squishy cross-check is owed for SCOPE-generality and ranks after the
discriminators (it tests corpus-generality, not the B-vs-b root).

## Bottom line

- **(B) or (b)?** Neither cleanly. NOT a separate prefetch-depth/out-of-order lever
  (those are dead/unfaithful). NOT a clean engine-W-root either: engine-W is proven
  root only at T3–T8 (91% busy); at T16-Ephys the engine-vs-feeding split is UNMEASURED
  and DIS-25's parallel-only 2x-E-core-extraction survives. DIS-26's "high-T DECODE-WAIT
  = the engine" is FIX-NEEDED — it imported the T3–T8 conclusion into the E-core regime
  without the E-core worker-idle measurement that would justify it, and folds a faithful
  per-chunk pipeline term (the 124ms@T1 oracle) into "the asm engine."
- **lean-consumer bounded?** YES — ≤23ms, correctly ceiling'd by the decompose bucket,
  consistent with DIS-16. SOUND.
- **output?** Shared floor, refuted as path. SOUND.
- **converged on engine-W-user-gated?** Partially: converged on the low-mid-T
  diagnosis; NOT on "only user decisions remain." Two un-refuted faithful items survive.
- **Highest-value next (a fix vs squishy vs user decision):** a MEASUREMENT, not a fix
  and not yet the user decision — extend the DIS-18 busy-fraction instrument to the
  T16-Ephys E-core mask and split DECODE-WAIT into worker-busy vs worker-idle (the
  cheap, decisive discriminator for THIS contradiction), and run the owed per-chunk /
  chunk-0 / ParallelSM-vs-direct removal oracle at T1 (isolate the 124ms gzippy-specific
  pipeline term). Same lever family. If both return "workers busy / pipeline
  irreducible" ⇒ converge on engine-W-user-gated, THEN squishy for scope, THEN hand the
  asm/bar decision to the user. If either shows idle E-cores or a removable pipeline term
  ⇒ a faithful work-distribution/pipeline lever is resurrected and must be attempted
  before the user decision.
