# TIER-2 AMENDED DIRECTIVE (supervisor ratification, 2026-06-06)

Supervisor + independent Opus advisor have reviewed the TIER-1 deliverable
(plans/tier1-design-and-tier2-proof.md). Full advisor disproof:
plans/tier1-advisor-verdict.md.

## RATIFIED
- The DIAGNOSIS (§0): rapidgzip ~0.46s is ~99% ISA-L; u16 ring-storage width
  refuted (vendor uses same 128KiB u16 ring); BMI2 rejected; vendor bench ceiling
  (igzip 720 vs best-pure 337 MB/s s-t, 1.28× at high-T); 128KiB inline dist cache
  + no shared live table. Advisor re-read the cited vendor lines and they check out
  (two cosmetic line-number slips noted, immaterial).
- The §1.1 governing-tension resolution: ONE no-FFI engine whose CLEAN inner loop
  is reimplemented igzip-class via pure-Rust + inline ASM (architecture stays
  faithful; inner Huffman loop is authorized open territory). This is the only way
  to honor BOTH the 1.0× bar and the one-engine/no-FFI governing memory. RATIFIED.

## MANDATORY AMENDMENTS BEFORE TIER-3 (advisor-required, supervisor-adopted)

1. **Run the CHEAP DECISIVE PRE-GATE FIRST — before PROOF-1/PROOF-2, before any
   SIMD build.** Per CLAUDE.md's own method (test the lever with a causal
   perturbation before a work-stretch):
   - **(a) Clean-loop slow-injection perturbation.** Slow ONLY the clean-mode
     inner loop by a known factor (GZIPPY_SLOW_BOOTSTRAP/ballast template) and
     measure the interleaved **T8 wall response** + a **frequency-neutral control**
     (sleep vs spin). Monotonic/proportional ⇒ clean-loop compute is on the
     critical path ⇒ proceed to PROOF-1. **FLAT ⇒ bandwidth/publish already binds
     ⇒ the AVX2 compute project is MOOT** — falsified in ~an hour instead of after
     a multi-week SIMD build.
   - **(b) u8-clean-write A/B.** Write clean bytes as u8 + skip the resolve pass
     for clean bytes, to isolate the TRAFFIC component (gap-B bandwidth). This
     directly tests amendment #2's live lever.
   Let THESE TWO experiments — not the design's priority list — rank SIMD-compute
   vs traffic/residency. Pre-register the falsifier for each before running.

2. **Re-frame CLAIM 2.** Ring-storage width is refuted (keep the 128KiB u16 ring —
   faithful, vendor-proven not the cause). BUT clean-bulk **write+resolve traffic
   width** (~100% of bytes streamed u16 through the ~20MiB chunk buffer; live u16
   write at marker_inflate.rs:1526) is a LIVE, unrefuted bandwidth lever in the
   exact T8 regime §0.3 calls bandwidth-sensitive. Promote u8-clean-write +
   drop-clean-resolve + dist-cache-shrink to FIRST-CLASS, separately-measured
   items. Do NOT pre-commit "don't shrink/keep-width" before the pre-gate measures.

3. **Fix PROOF-2 (the model).** Validating by reproducing the two walls it's tuned
   on is overfit-circular. Instead: **hold out** — fit on T1/T2/T4, PREDICT T8/T16
   as held-out, trust the faster-engine projection only if held-out lands within
   spread. AND read the binding term DIRECTLY from `perf stat` (MPKI/mem-stall +
   DRAM-ceiling probe, validated with the GZIPPY_MEM_BALLAST positive control) —
   the model is only a projector, the perf-stat measurement is the verdict.

4. **Strip absolute 337/720 MB/s targets** in favor of GUEST-measured ratios
   (reproduce vendor's own s-t ISA-L:pure ratio on the guest as PROOF-1's positive
   control / instrument self-test).

## SEQUENCE
PRE-GATE (1a+1b) → if compute-bound: PROOF-1 (toy decoder + ISA-L oracle control)
+ PROOF-2 (held-out model + perf-stat) → supervisor + advisor corroborate PROOF
RESULTS → only then TIER-3. If pre-gate is FLAT (bandwidth-bound), pivot the design
to the traffic/residency surface (amendment #2) and re-PROVE before TIER-3.
Structure-mandate work (dir split, dead-code removal) proceeds alongside, byte-exact.

## DISCIPLINES (unchanged, non-negotiable)
Serialize all builds via scripts/cargo-lock.sh; validate instruments with a
positive control first; numbers ONLY from the locked harness; dual-sha byte-exact
each step; reject a lever only with a mechanism; advisor corroboration before any
consequential/"done" claim.
