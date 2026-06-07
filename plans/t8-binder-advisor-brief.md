# DISPROOF ADVISOR BRIEF — is the T8 binder per-thread decode-compute, or the parallel-scheduling serial term?

You are an INDEPENDENT DISPROOF advisor. Your job is to BREAK the claim below, not ratify it.
Read the cited source first-hand. Reply with: UPHELD / UPHELD-WITH-CAVEATS / REFUTED, the
single strongest disproof attempt you made, and whether the campaign's next step should change.

## CONTEXT
gzippy is a pure-Rust parallel single-member gzip decoder, a faithful port of rapidgzip
(vendor/rapidgzip/). Goal: tie rapidgzip's whole-system interleaved wall. Current state: gzippy
~1.66x slower than rapidgzip at T8 on silesia (211968000 bytes decoded, sha-verified).

The campaign charter (plans/CAMPAIGN-CHARTER.md) currently asserts: "the gap to rapidgzip is a
CONSTANT ~1.70x at BOTH T1 and T8 => a PER-THREAD decode-throughput gap (decode BINDS), and the
inner Huffman LUT-decode kernel is the ONLY remaining lever to 1.0x." A prior round causally
established (via a faithful-u8 traffic perturbation) that clean-path TRAFFIC, ring width, and
PLACEMENT are slack, concluding the binder is per-symbol decode COMPUTE.

## THE NEW MEASUREMENT (locked guest, 16c, governor=performance, no_turbo=1, interleaved
sha-verified measure.sh, the project's only trusted harness). Instrument = slow_knob
(src/decompress/parallel/slow_knob.rs): a byte-transparent env-gated injection into the CLEAN-mode
inner decode loop (the production bulk path; GZIPPY_SLOW_HITS proves it fires 38.7M times = prop.
to clean decoded bytes). GZIPPY_SLOW_MODE=100 ~doubles per-thread decode-compute time;
GZIPPY_SLOW_KIND=sleep is the frequency-neutral control (real nanosleep, yields core, turbo-neutral).

CAUSAL PERTURBATION RESULT (reproduced across 3 runs, output sha-verified=OK every run):
- T1 (CPUS=0): OFF 0.533s -> spin100 0.974s = 0.548x => decode injection adds +83% wall.
  => At T1, per-thread decode-compute gates ~83% of the wall.
- T8 (CPUS=0,2,4,6,8,10,12,14): OFF 0.220-0.229s -> spin100 0.259-0.268s = +14% to +22% wall;
  spin200 -> +45% wall. Frequency-neutral sleep100 control: +20% (>= spin's +17%, so NOT a turbo
  artifact). => At T8, per-thread decode-compute gates only ~18-22% of the wall.
- Sensitivity ratio T1:T8 ~= 83% : ~20% ~= 4x. The SAME +100% decode-compute injection that adds
  +83% at T1 adds only ~20% at T8.

CORROBORATING TRACE (GZIPPY_VERBOSE, T8): decodeBlock aggregate 1.03s across 8 cores
(=0.129s/core ideal-parallel) vs wall ~0.22s (1.7x the ideal-parallel decode floor); Pool
Efficiency (Fill Factor) 73-83%; Prefetch dispatch saturated ~50/68; Worker resolve-ahead
ok=13/13; Early window publish handoff_key=13; 19 chunks.

## THE CLAIM TO BREAK
At T8, per-thread decode-compute is NOT the dominant binder (~18-22% of wall); ~78-82% of the T8
wall is a SERIAL / PARALLEL-SCHEDULING term (in-order publication, head-of-line stalls, pool
fill/dispatch saturation, last-chunk drain/CRC finalize) that OVERLAPS decode. Therefore the
inner-Huffman-kernel direction has a LOW T8 ceiling: even infinitely-fast decode removes at most
~18-22% of the T8 wall, leaving gzippy ~1.3-1.4x off rapidgzip at T8. The charter's "constant
1.70x = pure per-thread decode gap, kernel is the only lever to 1.0x" is WRONG AT T8.

## DISPROOF ANGLES YOU MUST TRY (find the strongest)
1. Is the T8 attenuation a SPARE-CORE artifact? CPUS pins 8 cores; box load was ~2.0-2.7 during
   runs (16c). Could idle sibling cores be absorbing the injected spin so the wall barely moves?
   (Counter: the sleep control yields the core and gives an EQUAL/larger rise; Pool fill 73-83%
   means workers ~busy. But scrutinize: does sleep truly remove the spare-core escape, given the
   injection is per-decode-event inside the worker?)
2. Does slow_spin>0 DISABLE the VAR_V fast loop (marker_inflate.rs:1478 gate
   `slow_spin == 0`)? It does — so the perturbation runs the CAREFUL loop. Does that bias the
   T1-vs-T8 contrast? (Both T1 and T8 OFF use VAR_V; both spin100 use careful. The CONTRAST
   between T1 and T8 is what's load-bearing, and both share the same OFF/ON mechanism, so the
   contrast should be robust to this. Verify the logic.)
3. Is the slope-to-ceiling inference valid? Rule 3 (CLAUDE.md): a slow-down slope proves "on the
   path" but NOT the speed-up ceiling. Here we infer a LOW ceiling from a LOW slope. Is "decode
   adds only ~20% wall when DOUBLED" sufficient to bound "speeding decode saves at most ~20%"? Or
   could a faster decode unbind something and pay more than the slope suggests? Argue both ways.
4. Reconcile with the prior round's "faithful-u8 traffic perturbation => decode COMPUTE binds at
   both T". Is the new T8 result consistent (compute binds T1, scheduling binds T8) or a
   contradiction that impugns one of the two instruments?
5. Spread: T8 spreads are wide (18-58%) and the +14-22% deltas sometimes fall inside spread
   (verdict "TIE"). Is the T1-vs-T8 CONTRAST (4x) real despite per-run TIE labels? (The contrast
   is reproduced 3x with consistent ordering OFF<spin100<spin200; the T1 +83% is far outside its
   ~6% spread.)

## SOURCE TO VERIFY (first-hand)
- src/decompress/parallel/slow_knob.rs (injection site, byte transparency, sleep control).
- src/decompress/parallel/marker_inflate.rs:1447-1478 (slow_spin gate, VAR_V fast-loop disable),
  :1638, :2083-2092 (injection call sites).
- src/decompress/parallel/chunk_fetcher.rs (dispatch saturation, prefetch horizon, in-order
  publication) — does the scheduling have a serial bottleneck a faster decode would NOT relieve?
- plans/CAMPAIGN-CHARTER.md (the claim being challenged), memory project_confirmed_offset_prefetch_gap
  ("high-T wall = head-of-line stalls, ~40% of T8 wall") — does this CORROBORATE the new finding?

## DELIVER
Verdict + strongest disproof + recommended next step (does the campaign re-point from the inner
kernel toward the T8 scheduling/serial term?). Be terse and decisive.
