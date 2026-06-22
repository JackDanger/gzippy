# AMD T≥2 vs rapidgzip — DEEP LOCATE: PRE-REGISTERED FALSIFIER

**Author:** AMD T≥2 locate leader. **Date:** 2026-06-22. **Branch:** `amd-t2t4-locate`
(off `origin/kernel-converge-A` @ `671c5752`). **Box:** solvency (AMD EPYC 7282 Zen2,
`root@10.0.2.240`). **Status:** committed+pushed BEFORE the first number is read
(CLAUDE.md governing law — a hypothesis carries its falsifier + the exact perturbation that
would test it, registered before measuring). This is the contract `AMD-T2T4-LOCATE-RESULTS.md`
is graded against.

## Starting point (GATED, from project_t2_rg_locate_2026_06_22 re-baseline)
Two distinct LOCATED SHAPES on AMD/Zen2 (Intel T≥2 essentially closed):
- **AMD-T4 = WORK-bound** (cyc≈wall): silesia-T4 cyc 1.081 ≈ wall 1.075; squishy-T4 cyc 1.087
  ≈ wall 1.075. gz retires ~8% MORE cycles than rg on the SAME chunks. PEXT/PDEP ruled out
  (objdump: gz 0 PEXT, 753 fast-BMI2; rg 0 PEXT). → genuine codegen/instruction-mix.
- **AMD-T2 = SERIALIZATION** (wall≫cyc): monorepo-T2 cyc 1.017 but wall 1.095. Pipeline not
  saturating cores. UN-LOCATED.

This cycle = DEEP GATED LOCATE of BOTH (no rewrite this cycle). No phases / infinite funding.

---

## SHAPE A — AMD-T4 WORK-bound inner-kernel cyc-excess locate

**Hot-path confirm (Gate-0/Gate-4 precondition):** perf-record (cycles) gz silesia-T4 (or
monorepo-T4) on Zen2 and confirm WHICH window-absent decode fn carries the cycles — the asm
`run_contig` clean kernel (`asm_kernel.rs:481`), the Rust `decode_marker_fast_loop`
(`marker_inflate.rs:2107`), `decode_clean_fast_loop` (:2478), or `decode_careful_tail`. Report
the symbol breakdown for gz AND rg (`Block::read`).

**HYPOTHESIS A:** the ~8% AMD cyc-excess lives in ONE dominant sub-region of the window-absent
decode kernel (a real gz-vs-rg codegen divergence — refill, table lookup, marker store/bookkeeping,
copy, bounds/branch, or register spill).

**Gate-2 perturbation (the verdict, NON-INERT required):** the NIGHT35 kernel injector
(`GZIPPY_KERNEL_INJECT` → `run_contig_inject`, asm_kernel.rs:1096) injects N dummy
instructions per hot-loop iteration into the clean asm kernel at T≥2. Self-validation: the
`note_exit`/asm-stats counter must fire (>0 run_contig calls) AND the injected wall must rise
monotonically+proportionally with N (freq-pinned cyc/B, /dev/null both arms, interleaved
N≥7) — proving the clean kernel is ON the T4 critical path. (If the marker fast loop is the
dominant symbol instead, build the analogous non-inert injection into `decode_marker_fast_loop`
and perturb THAT.)

**NAMES A DOMINANT CLOSEABLE BUCKET (GO) IF:**
- perf-annotate ranks ONE sub-region as ≳40–50% of the gz-minus-rg cycle excess, AND
- that sub-region is a real gz-vs-rg divergence (gz's asm spends materially more there than
  rg's Block::read on the same work), AND
- the NIGHT35 (or marker-loop) injector is NON-INERT and the wall responds
  monotonically+proportionally → the kernel is on the critical path (speeding it pays).
  → the open-territory inner-Huffman kernel codegen lever, named with the sub-bucket.

**NO-GO (report the actual shape, do NOT name a kernel rewrite lever) IF ANY:**
- the cyc-excess is DISTRIBUTED across the kernel (no sub-region ≳40–50%) — report "broad
  codegen, no single fat bucket".
- the dominant carrier of cycles is NOT the window-absent decode kernel (e.g. CRC, copy,
  alloc, apply_window) — re-attribute to that subsystem.
- the NIGHT35 injector wall is FLAT vs N (kernel has issue/latency slack at T4 — NOT
  work-bound on the critical path) — contradicts the cyc≈wall premise; report the contradiction.
- gz's hot symbol already matches rg's instruction mix (no divergence to close).

## SHAPE B — AMD-T2 SERIALIZATION decompose (monorepo-T2, wall 1.095 ≫ cyc 1.017)

**Decompose:** per-thread busy/idle duty via GZIPPY_TIMELINE (are all workers saturated? is
the in-order consumer thread the bottleneck? is the block-finder serial?), with conservation
(busy+idle==span; buckets reconcile to chunk count). Compare to rg's T2 core-saturation on the
same cell (is rg keeping cores busier?).

**HYPOTHESIS B:** ONE region serializes the AMD-T2 pipeline (workers idle waiting on it):
candidates = in-order consumer pace, serial block-finder, chunk-handoff/scheduling, or
prefetch depth. (The disproven-list has FALSE consumer-pace/placement levers — RE-LOCATE with
a firing perturbation; do not cite them.)

**Gate-2 perturbation (the verdict):** perturb the suspected serializing region by a KNOWN
factor (≥2×, frequency-neutral) and measure the interleaved T2 wall. If the wall moves
monotonically+proportionally → that region is on the critical path (serializer). If a worker
slowdown moves the wall but the consumer slowdown does not (or vice-versa) → discriminates
worker-bound vs consumer-bound. Non-inert: the injection counter must fire.

**NAMES THE SERIALIZER (GO) IF:** the timeline shows workers idle ≳40–50% of span waiting on
ONE region AND a ≥2× perturbation of that region moves the T2 wall proportionally while a
control perturbation of a non-suspected region does not. → faithful-rg pipeline convergence
target, cite vendor file:line.

**NO-GO (report the actual shape) IF ANY:**
- workers are ALL ≳90% busy (no idle) — then T2 is WORK-bound like T4, not serialization;
  re-classify (the cyc 1.017 then under-measured, or the wall≫cyc is a freq/turbo artifact —
  re-check freq-pinned).
- the idle is DISTRIBUTED (no single region ≳40–50% of worker idle).
- every perturbation moves the wall equally (no discrimination) — report "no isolable
  serializer; pipeline broadly under-saturated".
- rg shows the SAME core under-saturation at T2 (then it is not a gz-vs-rg divergence —
  inherent to the chunk-count/work-granularity at T2 monorepo).

## MEASUREMENT DISCIPLINE (applies to every number)
- **Gate-0:** rg comparator self-test (rg-vs-rg A/A ≈1.0±spread); every arm sha==zcat;
  /dev/null both arms; A/A ≪ Δ; every perturbation proven NON-INERT (counter fired).
- **Gate-1:** interleaved best-of-N≥7; report Δ AND inter-run spread; Δ<spread ⇒ TIE (label it).
- **Gate-2:** perturbation/removal-oracle is the ONLY verdict; freq-pinned cyc/B to separate
  work-bound vs serialization.
- **Gate-4:** GZIPPY_DEBUG=1 → path=ParallelSM + build-flavor=parallel-sm+pure; confirm the box
  binary built HEAD (sha grep / remote rev-parse — the push.default=tracking stale-build trap).
- **Box hygiene:** solvency freeze gov=performance + boost=0 (NOT no_turbo), idempotent +
  GUARANTEED restore + bounded auto-restore watchdog; NEVER leave frozen; verify+report
  gov=ondemand/boost=1 at exit. The user's `llama-completion` pegs 1 roaming core — taskset-pin
  measurement cores away from it; prefer T2/T4 (T8 untrusted last cycle); flag any neighbor
  perturbation (A/A spread). Do NOT touch bench-lock.sh.

**No negotiation after the fact.** A claim that did not pass every gate is an OPEN HYPOTHESIS,
labeled as such, not a finding.
