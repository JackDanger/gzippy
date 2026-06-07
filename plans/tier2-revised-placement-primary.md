# TIER-2 REVISED — PLACEMENT-PRIMARY (supervisor, 2026-06-07)

The pre-gate (d0aa1db / ce9fe6f, advisor-corroborated in plans/pre-gate-advisor-verdict.md)
REVISES the TIER-1 design's central premise. Recorded in memory
project_pregate_placement_is_dominant_lever.

## What the pre-gate proved
- Clean-mode inner Huffman COMPUTE is on the T8 critical path but its share is only
  **~11-29%** of the wall. Even *infinite* clean speedup → 0.79-1.00s, still loses to
  rapidgzip ~0.53s. So **class-C SIMD is bounded-secondary, NOT the lever to the tie.**
- **The dominant lever is PLACEMENT / head-of-line stalls (~58% of T8 wall; ~42%
  parallel efficiency)** = the confirmed-offset-prefetch-gap. Placement is ARCHITECTURE,
  so the FAITHFUL-PORT mandate applies: port rapidgzip's chunk scheduling/prefetch/
  block-finder confirmation — do NOT innovate here (unlike the inner loop).

## REVISED SEQUENCE (still tiered: PROVE ceilings before any work-stretch)

**STEP A — Bound BOTH ceilings with REMOVAL ORACLES (CLAUDE.md rule 3: slow-down slope
≠ speed-up ceiling). No design commitment until these land.**
  - **Oracle-C (clean compute):** remove/zero the clean-decode time (e.g. decode-once-
    then-replay, or skip-decode-emit-known-bytes oracle) and measure the T8 wall. This
    RESOLVES the 11-29% bracket and sets the hard class-C ceiling. Must be byte-exact-
    bypassed or clearly marked non-producing (it's an oracle, not production).
  - **Oracle-P (placement):** the decisive one. Construct a perfect-placement / zero-
    head-of-line-stall oracle (e.g. feed workers the CONFIRMED offsets so no chunk waits
    on a mispredicted partition boundary; or an idealized scheduler that never stalls the
    consumer's frontier) and measure the T8 wall. If perfect placement → ~0.47-0.53s,
    **placement is the proven path to the 1.0× tie.** If it lands well short, the bar
    needs additional levers (traffic, etc.) — report it; the proof, not optimism, decides.
  - Pre-register a falsifier for each oracle BEFORE running; validate each instrument with
    a positive control; numbers ONLY from the locked guest harness; interleaved N≥9.

**STEP B — Experiment 1b (traffic/residency A/B), still owed.** u8-clean-write + drop-
  clean-byte resolve over the ~20MiB u16 buffer, to rank class-T as a co-lever (the
  advisor's live-but-unmeasured bandwidth asymmetry). Bound its ceiling too.

**STEP C — REVISE the TIER-1 design** around the measured ranking: PLACEMENT primary
  (faithful port of rapidgzip's scheduler/prefetch), class-T and class-C as bounded
  secondaries justified by their oracle ceilings. Deliver the revised design + the three
  oracle ceilings to supervisor + independent advisor.

**STEP D — only after corroboration: TIER-3 align** (placement first), dual-sha byte-exact
  every commit (028bd002...cb410f both features), wall verified on the locked harness.

## Structure mandate (still owed, runs alongside, byte-exact)
gzippy-isal / gzippy-native subdir split; remove dead code (unified.rs HAS_BMI2 placeholder;
specialized_decode/SPEC_CACHE production-dead cluster; fix stale isal-compression engine-A/B
comment guest_fulcrum_capture.sh:69-71). Names describe behavior. The native/isal clean-path
confusion that mis-sited the slow_knob (resumable=isal vs marker_inflate=native) is EXACTLY
why this split matters — it would have prevented that defect.

## DISCIPLINES (unchanged)
Run subagents SYNCHRONOUSLY (no false auto-reinvoke); NO detached sleep sentinel (supervisor
enforces single-leader); serialize builds via cargo-lock.sh; verify the guest is IDLE before
each measurement (orphaned harness runs have skewed it); reject a lever only with a mechanism;
advisor corroboration before any consequential claim.
