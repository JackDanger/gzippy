# Pre-gate advisor verdict — clean-loop COMPUTE criticality (T8)

Advisor: independent disproof-driven Opus. READ-ONLY. HEAD d0aa1db, branch reimplement-isa-l.
Verdict under attack: *"clean-mode inner Huffman COMPUTE is on the T8 critical path
(COMPUTE-BOUND), class-C SIMD-compute NOT moot; proceed to PROOF-1."*

Source verified, not just the summary: slow_knob.rs (full), marker_inflate.rs:1291-1307 &
1534-1546 (both clean arms, `CONTAINS_MARKERS=false`, const-folded on marker path, inject at
top of the per-decode-event outer loop before refill), gzip_chunk.rs:1205-1220 (native fold,
Engine M continues), run_locked_fulcrum.sh (harness, N=9 interleaved, dual-sha).

## (A) Bottom line: **CORROBORATE-WITH-CAVEATS**

The narrow claim survives: clean-mode inner-loop compute is **on** the T8 critical path
(monotonic slope, survives the frequency-neutral sleep control ⇒ not a pure turbo artifact ⇒
class-C is **not literally moot**). What does **not** survive is the *label* "COMPUTE-BOUND"
and any reading of "proceed to PROOF-1" as "begin the SIMD build." The slow-down slope only
licenses "not-moot"; the speed-up ceiling is still unbounded and PROOF-1 (removal oracle) is
**mandatory before any work-stretch** — and the bracket the slope already gives makes a large
payoff unlikely.

## (B) Strongest disproof I could mount — and whether it survives

**Disproof attempt:** "The sleep slope is a scheduling/oversleep artifact, not compute
criticality. Batched `thread::sleep(50µs)` deschedules a worker; under T8 load nanosleep
overshoots; the added wall is descheduling latency / consumer-wait perturbation, not the clean
loop being on the path. The harness itself says consumer wait = 774ms = 77%, RATE-dominant —
i.e. the consumer is *starved*, so the wall is set by something upstream that may not be the
clean compute."

**Does it survive? NO — it actually CONFIRMS the narrow claim.** A descheduled or
over-slept worker only moves the wall *if that worker's output is what a starved consumer is
waiting on* — i.e. only if worker production is rate-limiting and the perturbed region sits on
the serial critical path. If clean decode were slack, descheduling during it would be absorbed
and the wall would be FLAT. It is not flat; it is monotonic (sleep 1.7→7.0→10.7%). So the
consumer-wait=77% / RATE-dominant observation and the slope are the **same finding**: workers
are the rate limiter and clean compute is one of the binding inputs. Attack #1 fails to refute
"on the path." (Oversleep, if present, makes the sleep slope an UPPER bound on injected wall ⇒
true clean share ≤ the sleep number, which only tightens caveat C1, never reaches FLAT.)

I could NOT construct a path to FLAT / bandwidth-bound (attack #5). The spin loop touches only
a register accumulator (no array) ⇒ negligible cache/bandwidth footprint, so this is not a
bandwidth probe and there is no bandwidth verdict to extract either way.

## (C) Caveats / required next steps — the real problem is MAGNITUDE, not direction

**C1 — The two slow-down methods DISAGREE, and an independent cross-check sides with the
LARGER one. PROOF-1 is non-negotiable.**
- Sleep (freq-neutral floor): +10.7% at F=100 ⇒ clean share ≈ 11% of T8 wall.
- Spin: +28.2% ⇒ ≈ 28%.
- **Independent cross-check from the T1 positive control:** T1 F=100 spin = +71%, sd 0.1%,
  and T1 has no all-core turbo confound (one core boosts regardless). +71% ⇒ clean compute ≈
  0.71 × 3.734s = **2.65s of single-thread CPU work**. Spread ideally over 8 cores = 0.33s;
  against the 1.121s T8 wall that is **~29%** — i.e. the T1 extrapolation *matches the SPIN
  slope (28%), not the sleep slope (11%)*. So the usual "spin is turbo-inflated, trust sleep"
  reflex is suspect here: spin coincides with a turbo-clean independent estimate, and the
  batched-sleep number is plausibly *under*-reporting (a yielded core lets other work overlap).
- Net: the slope brackets clean's share at **roughly 11–29%** with the methods in genuine
  tension. Per CLAUDE.md rule 3, slow-down slope ≠ speed-up ceiling regardless. **The removal
  oracle (PROOF-1) is the only instrument that resolves the bracket and bounds the ceiling. Do
  not start the SIMD implementation on the strength of this pre-gate.**

**C2 — Even the optimistic end cannot close the rapidgzip gap; class-C is not the T8 lever.**
gzippy T8 = 1.121s, rapidgzip ≈ 0.53s (loss ratio 0.47, ~2.1×). Infinite speed-up of clean
compute (the unreachable upper bound) removes at most its share: at 29% → 0.79s; at 11% →
1.00s. **Both still lose to rapidgzip by a wide margin.** Realized SIMD gains (a 2× clean-loop
speed-up is already aggressive) buy ~5–14% wall, and rule 3 says the next component binds
before even that. Meanwhile the box is far from compute-saturated: ideal T8 = 3.734/8 ≈ 0.47s
vs actual 1.121s ⇒ **~42% parallel efficiency, ~58% lost to serialization/placement**, which is
exactly the [[project_confirmed_offset_prefetch_gap]] head-of-line story (~40% of T8 wall from
4 confirmed-offset stalls). **The dominant T8 lever is placement/head-of-line, not clean
compute.** "Not moot" ≠ "high-value." Class-C should be sequenced AFTER the placement lever, or
explicitly justified as a structural-parity port rather than a wall lever.

**C3 — Scope the conclusion to "compute is one on-path input with a modest share," and DROP the
"COMPUTE-BOUND" label.** RATE-dominant + 77% consumer wait + 58% idle/stall describes a wall
that is *mixed* and at T8 *more* placement/serialization than compute. Recording this cell as
"COMPUTE-BOUND" is the exact overstatement that would wrongly license a multi-session SIMD
work-stretch. Record it as: "clean compute on the critical path, share ~11–29% (methods
disagree; removal-oracle pending), gap-closing lever = placement."

## (D) Instrument / site concerns

- **Site is valid.** Injection is in the `CONTAINS_MARKERS=false` specializations only,
  const-folded to 0 on the marker path, in BOTH clean arms, once per outer decode event (= one
  Huffman codeword → 1–3 literals or one back-ref) before refill/decode. That is the correct
  per-decode-event compute unit. No double-count: the two arms are alternative specializations,
  one per decode event, and the 40.13M hit count being ∝ bytes (~5 B/event, consistent with
  multi-literal + back-ref emit) rules out 2× firing.
- **Re-assert the counter on the perturbed build.** The 40.13M validation was a ~203MB decode;
  the perturbation runs are silesia-large 503MB at T8. Same path, but re-run `GZIPPY_SLOW_HITS=1`
  on the *exact* 503MB T8 build to make the site-fires-∝-clean-bytes proof airtight for THIS
  experiment, not a smaller sibling run.
- **Sleep batching (50µs) can oversleep under T8 load** ⇒ sleep slope is an upper bound on
  injected wall ⇒ if anything the true clean share is ≤ the sleep figure; cannot manufacture a
  false positive (still requires worker-rate-limiting to move the wall). Noted, not blocking.
- **Calibration is single-thread (Rosetta x86 T1), applied at T8.** `NS_PER_SPIN_ITER=0.32`
  and `BASE_SPIN=22` were fixed at T1; per-event compute cost differs under T8 cache/SMT
  pressure, so the F→%wall mapping is approximate. Fine for monotonicity (the only thing the
  pre-gate needs); do NOT read the F=100 percentages as precise shares — that is PROOF-1's job.

## One-line instruction to the requester

Promote the verdict to **"clean compute is ON the T8 critical path (class-C not moot)"** and
proceed to **PROOF-1 = run the bootstrap/clean removal oracle to BOUND the speed-up ceiling**
— NOT to implementing SIMD. Carry forward two facts the oracle must beat: (1) the slope already
brackets clean's share at ~11–29% with spin≈T1-extrapolation in tension with sleep, and (2)
even the 29% end leaves T8 ≈0.79s vs rapidgzip 0.53s, so the placement/head-of-line lever
([[project_confirmed_offset_prefetch_gap]]) outranks class-C for closing the rapidgzip gap.
Strike the "COMPUTE-BOUND" label.
