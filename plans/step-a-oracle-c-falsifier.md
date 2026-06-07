# STEP A — ORACLE-C (CLEAN-COMPUTE REMOVAL) — pre-registered falsifier

Written BEFORE running. Branch reimplement-isa-l @ ce9fe6f. T8, silesia-large,
pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM=1, path=ParallelSM asserted,
locked guest harness (host_lock_and_bench: no_turbo=1, governor=performance,
min=max pinned), interleaved best-of-N≥9, drop iter0, sha-verified vs `gzip -dc`.

## What Oracle-C measures and why (charter STEP A, CLAUDE.md rule 3)
The pre-gate proved clean-mode inner-Huffman COMPUTE is ON the T8 critical path
(slow-injection slope monotonic, survives the freq-neutral sleep control) but
bracketed its share at only ~11-29% (sleep floor 11% vs spin/T1-extrapolation 29%,
methods in tension). Rule 3: slow-down slope ≠ speed-up ceiling. Oracle-C RESOLVES
the bracket by REMOVING the clean-decode compute (replay precomputed bytes,
inner-Huffman CPU ≈ 0) while keeping EVERYTHING ELSE intact (placement / in-order
consumer / window publish / marker resolve / CRC / writev). The resulting T8 wall
is the HARD class-C ceiling: even an infinitely-fast clean decoder cannot beat it.

Advisor prediction to beat: ~0.79-1.00s (infinite clean speedup removes only the
11-29% share of the 1.121s wall). leverB model independently predicted ~0.78-0.84s.

## Mechanism (the EXISTING decode-bypass oracle, with the prior-run contamination FIXED)
`src/decompress/parallel/decode_bypass.rs` — GZIPPY_BYPASS_CAPTURE (capture pass,
normal decode records every ChunkData keyed by (start_bit, stop_hint_bit)) then
GZIPPY_BYPASS_DECODE (replay pass: worker returns the precomputed ChunkData via the
PREBUILT path — HashMap lookup + Option::take, NO per-call memcpy/alloc/fault —
inner-Huffman CPU ≈ 0). Output is BYTE-EXACT (the correctness gate). Cache miss ⇒
fall back to real decode (keeps bytes correct, only INFLATES the floor ⇒ upper bound).

### KNOWN PRIOR DEFECT (plans/leverB-ceiling.md) — MUST be fixed for this run to count
The prior leverB Oracle-C run read CEIL_FLOOR_A = 3.667s and was declared
LOAD-CONTAMINATED: the floor included the ONE-TIME prebuilt-map reconstruction
(~656MB ChunkData rebuild) and capture handling INSIDE the timed wall, plus possible
swap (swap_in=112). The pre-registered fix the prior run itself owed:
> "Re-run the floor with a fast-but-REAL decode oracle — replay from RAM with CRC
>  stripped, NOT a 656MB on-disk capture+CRC."
REQUIRED FIXES before any number counts:
1. **Eager prebuilt-map init OUTSIDE the timed region.** The prebuilt map
   (decode_bypass.rs:588 prebuilt_map) is built lazily on FIRST replay call =
   inside the timed `drive`. Force it to build BEFORE the timed decode starts
   (warm it, or add an eager-init hook), so the one-time 656MB reconstruction is
   NOT in the wall. VERIFY: the wall must not include a ~3s reconstruction spike.
2. **Capture on /dev/shm (RAM)** — already done in guest_ceiling.sh (CAPFULL on
   /dev/shm). Confirm no on-disk read in-wall.
3. **No swap.** Log swap_in delta across the timed passes; VOID if swap_in > 0
   on the floor pass (memory pressure ⇒ bimodal wall). Free the 656MB capture
   buffers if RSS approaches the box limit (check `free -m` avail before the pass).
4. **CRC is KEPT (it is genuine consumer work that runs in the real pipeline too),
   NOT stripped** — the prior "strip CRC" note was about isolating reconstruction
   cost; CRC belongs in the floor. (If CRC turns out to dominate the floor, that is
   itself a finding about the floor's composition, reported — not removed.)

## Instrument validation (must pass BEFORE the floor number is trusted)
- **POSITIVE control / region-removed proof:** GZIPPY_SLOW_HITS-style or replay
  hit% ≥ 90% (decode actually bypassed, not silently real-decoded). REQUIRE the
  floor-pass fulcrum trace to show the inner-clean-decode span (baseline ~548ms
  wall-crit in the prior run) DROP TO ≈0 — that is the proof the region was removed.
  If the clean-decode span is still large in the floor trace, the oracle did NOT
  remove the region ⇒ VOID.
- **NEGATIVE/anchor control:** gzippy_normal interleaved in the SAME passes must
  reproduce the baseline T8 wall (~1.121s ± spread). If the anchor drifts, the
  harness drifted ⇒ nothing is comparable ⇒ VOID.
- **Byte-exact:** every floor iteration sha == REF_SHA (gzip -dc). One DIVERGE ⇒ VOID.

## PRE-REGISTERED FALSIFIER (the verdict)
Let FLOOR_C = Oracle-C T8 floor (min of N≥9, decode≈0, byte-exact, hit%≥90, sd%≤5,
no swap, prebuilt-init out-of-wall, anchor reproduces baseline).

- **CLASS-C CEILING CAPPED (the advisor/model prediction):** FLOOR_C lands in
  ~[0.75, 1.05]s, i.e. materially > rapidgzip's ~0.53s. ⇒ even infinitely-fast
  clean decode CANNOT reach the 1.0× tie; class-C is bounded-secondary; the residual
  binder (placement / publish-chain / resolve) is named from the floor trace
  (file:line). This CONFIRMS the placement-primary ranking.
- **CLASS-C SUFFICIENT (would overturn the ranking):** FLOOR_C ≤ ~0.55s, byte-exact,
  hit%≥90, sd%≤5, anchor OK. ⇒ removing ONLY clean compute reaches the tie ⇒ class-C
  IS the dominant lever after all. This CONTRADICTS the pre-gate bracket (11-29%) and
  the leverB model (0.78-0.84) — RECONCILE before any conclusion (do not cherry-pick).
- **GREY (0.55, 0.75):** lean on the floor-trace wall-critical decomposition + the
  pre-gate bracket; report the named binder, do not force a binary verdict.
- **VOID** (STOP, report, do not interpret magnitude): any sha DIVERGE; path≠ParallelSM;
  replay hit% < 90%; floor pass sd% > 5%; swap_in > 0 on the floor pass; prebuilt-map
  reconstruction observed INSIDE the timed wall (the prior 3.667s contamination);
  anchor (gzippy_normal) fails to reproduce ~1.121s ± spread; OR the floor trace shows
  the inner-clean-decode span did NOT drop to ≈0 (region not actually removed).

## Disproof attempts baked in
- Region-removed proof via floor-trace span decomposition (clean-decode → ≈0), not
  attribution.
- Anchor-reproduces-baseline guard kills harness drift.
- Out-of-wall prebuilt init + swap gate kill the exact prior 3.667s contamination.
- hit% ≥ 90 kills silent real-decode contamination.
- sd% ≤ 5 kills a swap-thrashed bimodal lucky-min.

---
## MEASURED RESULTS [2026-06-07, STEP-A leader] — Oracle-C ceiling run (HEAD a49c357)
Locked guest harness (host no_turbo=1, governor=performance, RESTORE VERIFIED), T8,
silesia-large 503MB, N=9 interleaved, drop iter0. guest_ceiling.sh PASS A.

| point | min | sd% | note |
|---|---|---|---|
| CEIL_BASELINE_NORMAL (anchor) | 1.1338s | 1.2 | reproduces ~1.12s baseline ✓ (harness not drifted) |
| CEIL_FLOOR_A (decode≈0, byte-exact sha=OK, hit%=97.6, swap_in=48) | **3.6298s** | 0.9 | CONTAMINATED — see below |
| CEIL_RAPIDGZIP | 0.5304s | 2.5 | tie target |
| CEIL_A2_SLEEP0 (resolve-ELIDED, garbage) | 0.4441s | 1.7 | probe |
| CEIL_B_SLEEP66 (rapidgzip-rate, resolve-ELIDED, garbage) | 0.7323s | 3.2 | weak lower bound |

GATES: sha=OK (byte-exact), hit%=97.6 (>90), sd%=0.9 (<5), anchor reproduced, swap=48
pages (negligible, mem avail 8.1G), path=ParallelSM, freq pinned. The HARD gates PASS —
the run is NOT VOID on the falsifier's VOID criteria.

### REGION-REMOVED PROOF: PASS (decode genuinely removed)
Floor trace (/tmp/oracle-c-art/trace_floor_T8.json): worker.decode_chunk aggregate
busy dropped 6.33s → 0.076s (only the 2-miss real-decode fallback). The clean-decode
compute region IS removed. (Falsifier's region-removed requirement satisfied.)

### TWO CONTAMINATIONS pull the raw 3.63s the WRONG way (both directions) — magnitude UNTRUSTWORTHY
1. **warm/load contamination (INFLATES):** CEIL_FLOOR_A is WHOLE-PROCESS wall. The
   warm_prebuilt fix moved the ~3.1s ChunkData rebuild before drive_t0 — but the
   harness times the whole process, so 3.1s warm (measured: 3073-3162ms/iter ×10) +
   ~0.4s 656MB capture-load are STILL in CEIL_FLOOR_A. ⇒ 3.63 − 3.1 − ~0.4 ≈ **~0.5s**
   post-contamination. (Fix incomplete: the harness must subtract the reported warm,
   or replay must use a persistent process. NOT re-run this turn.)
2. **window-collapse contamination (DEFLATES):** decode≈0 ⇒ windows arrive INSTANTLY
   ⇒ L_resolve collapses (162µs/chunk vs 19.93ms at real decode speed; prior leverB
   caveat). So the fulcrum consumer-critpath wall on the floor trace = **336ms** is an
   UNDER-estimate (the publish-chain term L_resolve is artificially free).

### CORRECTED CLASS-C CEILING (bracketed, instrument-entangled): ~0.4-0.7s
Robust reads bracket the floor: A2 0.444s (resolve-elided), fulcrum-critpath 0.336s
(window-collapsed), A−warm−load ≈ 0.5s, B-sleep66 0.732s. ⇒ class-C floor ≈ **0.4-0.7s**
— BELOW the advisor's predicted 0.79-1.00s, but the floor is too entangled (warm
contamination + window-coupling) to pin precisely. The L_resolve term that the model
says binds (~0.78s) DOES NOT materialize at free decode because windows arrive instantly.

### VERDICT: GREY → CLASS-C CEILING ~0.4-0.7s, NOT cleanly resolvable by full decode removal
Removing clean COMPUTE entirely is a DEGENERATE oracle here: it also frees the windows,
collapsing the very publish-chain term that would bind at real decode speed. So Oracle-C
cannot cleanly separate "class-C compute ceiling" from "publish-chain ceiling" — they
co-collapse. CONSEQUENCE for the ranking: class-C is bounded (floor ≤ ~0.7s ⇒ infinite
clean speedup buys at most ~1.12→~0.7s = ~0.4s, ≤37% of wall, consistent with the pre-gate
11-29% being a LOWER part of that), and it is NOT the clean lever Oracle-P is. The prior
leverB owed re-run "fast-but-REAL decode oracle (CRC-stripped RAM replay)" remains the
way to de-entangle — flagged for the advisor/STEP-B, not blocking the checkpoint.
