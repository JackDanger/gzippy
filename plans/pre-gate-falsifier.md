# PRE-GATE FALSIFIER — pre-registered BEFORE running (amendment #1)

Charter: plans/tier2-amended-directive.md (amendment #1, the cheap decisive
pre-gate). This document is written and committed BEFORE any measurement, per
CLAUDE.md PROCESS rule 5 (disproof-driven, state the claim then try to break it)
and the advisor verdict (plans/tier1-advisor-verdict.md, "run it FIRST").

The pre-gate decides which lever class to invest in:
- **class-C (SIMD compute)** — igzip-class vectorized clean inner loop (weeks, high-risk)
- **class-T (traffic/residency)** — u8-direct clean writes, drop clean-byte resolve, dist-cache shrink (hours, low-risk)

Two experiments, each with a pre-registered falsifier. Numbers ONLY from the
locked Fulcrum harness (scripts/bench/run_locked_fulcrum.sh / guest_fulcrum_capture.sh),
interleaved best-of-N≥7, sha-verified (028bd002…cb410f), path=ParallelSM asserted,
on the frozen build guest. ALL builds via scripts/cargo-lock.sh.

---

## EXPERIMENT 1a — clean-loop slow-injection causal perturbation

**Claim under test:** the clean-mode inner loop's COMPUTE is on the T8 critical
path (the design's whole premise — "clean-loop compute is the T8 lever").

**Mechanism (the instrument, to be built byte-transparently):** a SLOW knob that
adds a known, controllable amount of work INSIDE the clean-mode inner loop ONLY
(`read_internal_compressed_canonical_specialized::<false>` and
`read_internal_compressed_specialized::<false>` in
src/decompress/parallel/marker_inflate.rs — the `CONTAINS_MARKERS=false` arm,
where ~99% of bytes are decoded post-flip). Gated on an env var, compiled in but
a no-op when the var is unset (early-return on a OnceLock<bool>, like mem_stats),
so the OFF path is byte- AND perf-identical to today's production binary (proven
by dual-sha + a baseline TIE of OFF vs the current HEAD binary).

  - The injected work is proportional to the loop's OWN measured work (e.g. a
    busy-spin / sleep of N% of per-iteration or per-symbol cost), so a factor F
    means "the clean loop now costs ~(1+F)× its compute."
  - TWO variants, MANDATORY (CLAUDE.md PROCESS rule 2, frequency-neutral control):
    - **SPIN** (busy-loop) — can depress all-core turbo, inflates the delta.
    - **SLEEP** (yields the core) — frequency-neutral control. If the delta
      SURVIVES under SLEEP, the criticality is real, not a turbo artifact.

**Instrument self-test (POSITIVE CONTROL — run BEFORE trusting any verdict,
CLAUDE.md PROCESS rule 4):**
  1. OFF (knob unset) vs current-HEAD binary: dual-sha identical AND T8 wall TIE
     (Δ < spread). Proves the knob is byte/perf-transparent when off.
  2. A KNOWN-LARGE injection (e.g. F=2.0, i.e. +200%) at T1 (single-thread, where
     the clean loop is unambiguously serial and on the path) MUST move the T1 wall
     monotonically and by a large, clearly-out-of-spread amount. If even a +200%
     injection at T1 does NOT move the T1 wall, the knob is not actually slowing
     the loop — the instrument is broken; fix before trusting any T8 number.

**PRE-REGISTERED FALSIFIER / verdict rule (decided BEFORE the run):**
  - Inject at F ∈ {0, 0.25, 0.50, 1.00} (a slope, not one point). Measure the
    interleaved T8 wall at each F, best-of-N≥7, both SPIN and SLEEP.
  - **MONOTONIC & PROPORTIONAL T8 response** (T8 wall rises with F, out of spread,
    and the SLEEP variant preserves the rise) ⇒ **clean-loop compute is on the T8
    critical path ⇒ COMPUTE-BOUND ⇒ proceed to PROOF-1 (the SIMD-compute proof).**
  - **FLAT T8 response** (T8 wall Δ < inter-run spread across F up to +100%, while
    the T1 positive control confirms the knob really slows the loop) ⇒ **the clean
    loop's compute is SLACK at T8 ⇒ bandwidth/publish already binds ⇒ the AVX2
    compute project is MOOT.** Pivot to class-T (traffic/residency) and re-PROVE
    before TIER-3.
  - **TIE / ambiguous** (Δ ≈ spread, or SPIN moves but SLEEP doesn't) ⇒ NOT a
    clean compute-bound verdict; treat as FLAT-leaning (turbo artifact suspected),
    report both numbers, do NOT proceed to the SIMD build on a spin-only delta.

The decisive comparison is the SLOPE of T8-wall-vs-F, not any single absolute.

---

## EXPERIMENT 1b — u8-clean-write A/B (isolate the traffic component)

**Claim under test:** the clean-bulk write+resolve TRAFFIC width (gzippy writes
u16 per literal into the ring on ~100% of bytes at marker_inflate.rs:1526, and
re-streams the full chunk u16 buffer in the resolve pass) is a LIVE T8 bandwidth
lever — the amendment #2 "gap-B traffic" component, distinct from SIMD compute.

**Mechanism (the A/B):** a variant of the clean-mode path that writes clean bytes
as **u8 directly** (the ring is already u8-interpretable post-flip) AND skips the
u16 resolve pass for clean bytes (markers exist only in the ≤32 KiB prefix, so the
clean bulk needs no resolve). This is a real behavioral change — it MUST remain
byte-exact (dual-sha 028bd002…cb410f) or the variant is void.

  - If a fully byte-exact u8-clean-write variant is too invasive to land safely
    inside the pre-gate window, the FALLBACK isolation is a TRAFFIC-only
    perturbation that is byte-exact: add a known volume of EXTRA u16 ring
    traffic on the clean bulk (a redundant second write of each clean symbol to
    a throwaway scratch ring of the same size) and read the T8 wall response —
    same logic as 1a but perturbing TRAFFIC, not compute. Monotonic ⇒ traffic is
    on the path. This keeps output bytes identical (the scratch write is dead).
    The PRIMARY (real u8-write) A/B is preferred when achievable byte-exact; the
    traffic-injection fallback is the byte-safe substitute that still isolates the
    traffic term.

**Instrument self-test (POSITIVE CONTROL):** the traffic-injection fallback at a
large extra-traffic factor at T8 MUST move the wall if and only if traffic is on
the path; pair with a T1 control to confirm the extra writes actually execute
(T1 wall must rise — extra memory writes always cost something single-thread).

**PRE-REGISTERED FALSIFIER / verdict rule:**
  - **u8-write A/B shows a T8 wall REDUCTION out of spread (byte-exact)** ⇒ the
    clean-bulk traffic is a live T8 lever ⇒ class-T is real, promote it.
  - **Traffic-injection shows MONOTONIC T8 wall increase with extra u16 traffic**
    ⇒ traffic is on the T8 path (corroborates class-T) even if the full u8-write
    A/B isn't landed in-window.
  - **FLAT** (extra traffic does NOT move T8, T1 control confirms writes execute)
    ⇒ traffic is NOT the T8 binding term; class-T traffic lever is de-ranked
    (residency/MPKI may still bind — that is PROOF-2/perf-stat territory).

---

## COMBINED PRE-GATE VERDICT MATRIX (decided before running)

| 1a (compute) | 1b (traffic) | Verdict |
|---|---|---|
| MONOTONIC | any | COMPUTE-BOUND → proceed to PROOF-1 (and keep class-T as co-lever if 1b also moves) |
| FLAT | MONOTONIC/u8-win | BANDWIDTH/TRAFFIC-BOUND → pivot design to class-T, AVX2 project MOOT, re-PROVE |
| FLAT | FLAT | NEITHER compute nor traffic binds → the binding term is residency/publish/something else → escalate to PROOF-2 perf-stat (MPKI/mem-stall) BEFORE any build; report to supervisor |
| MONOTONIC | MONOTONIC | BOTH bind → compute-bound primary, traffic co-lever; PROOF-1 + class-T both in scope |

**Δ < inter-run spread ⇒ TIE ⇒ treated as FLAT for that axis (CLAUDE.md rule 5).**

## DISCIPLINES (binding on this pre-gate)
- Build the SLOW/traffic knob byte-transparently (OFF == today, dual-sha proven).
- Validate each instrument with its positive control FIRST; a knob that doesn't
  move the T1 wall under a large injection is BROKEN — fix before any T8 verdict.
- Numbers ONLY from the locked Fulcrum harness; interleaved best-of-N≥7; frozen
  guest; SPIN+SLEEP for 1a (frequency-neutral control).
- Serialize ALL builds via scripts/cargo-lock.sh; ONE build at a time; check df -h.
- The verdict reshapes everything downstream → CHECKPOINT to supervisor + advisor
  with the numbers BEFORE the expensive PROOF-1/PROOF-2.
