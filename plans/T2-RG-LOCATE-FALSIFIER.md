# T≥2 vs rapidgzip — RE-BASELINE + FRESH LOCATE: PRE-REGISTERED FALSIFIER

**Author:** T≥2-front leader. **Date:** 2026-06-22. **Branch:** `t2-rg-locate`
(off `origin/kernel-converge-A` @ `036b835d`). **Status:** pre-registered BEFORE any
number is read (CLAUDE.md governing law — a hypothesis must carry its falsifier and the
exact perturbation that would test it, committed before measuring).

This doc is committed+pushed BEFORE the first measurement. It is the contract the
results (`plans/T2-RG-LOCATE-RESULTS.md`) are graded against.

---

## The question this cycle answers (and ONLY this — no port this cycle)

1. **Gate-0:** does a rapidgzip native ELF exist on EACH box (neurotic Intel, solvency
   AMD) and self-test rg-vs-rg ≈ 1.0 ± spread? (The "no rg on solvency" / stale-RG_BIN
   trap has burned the campaign.)
2. **Re-baseline:** at the CURRENT pinned binary (`kernel-converge-A` HEAD, which DOES
   contain the view-list convergence `300e772b` — verified `is-ancestor` = true), what is
   the gz-native/rg wall matrix at T2/T4/T8 on silesia+nasa+monorepo+squishy, BOTH arches?
   **Which cells does gz LOSE (gz/rg slower than 0.99, i.e. rg_wall/gz_wall > 1.01)?**
3. **Fresh gated LOCATE:** on a confirmed losing cell, what fraction of the tax is the
   speculative u16-MARKER / apply-window / replace-markers machinery vs the shared
   parallel pipeline (block-finder, chunk lifecycle, consumer pace) vs decode? Ranked,
   via a Gate-2 removal-oracle (the ONLY verdict), with non-inert proof (counter fired /
   hits==expected).

---

## PRE-REGISTERED HYPOTHESES + their falsifiers

### H-BASELINE — "the T≥2 gap reappears as a real loss at HEAD"
- **Predicts:** ≥1 T≥2 cell with rg_wall/gz_wall > 1.01 (gz loses), Δ > inter-run spread,
  replicated (or at least directionally consistent) on both arches.
- **FALSIFIED IF:** all T≥2 cells are ≥ 0.99 (TIE-or-win) on both arches → the prime-
  directive T≥2 gap is CLOSED at HEAD; the cycle's deliverable becomes "confirm + the
  remaining debt is AMD-LAW / T1," and there is NO losing cell to locate. (Per the
  T2-LOCATE-PLAN FORK-A, this is the *likely* state on Intel post-view-list.)
- **NOTE (anti-amnesia):** the banked record says Intel reached instruction-parity at T4
  via `300e772b`, with a residual **silesia-T4 wall ~1.15** declared "at the in-order-tail
  structural floor" (prefetch-depth A/B FALSIFIED). So silesia-T4 is the most likely
  surviving Intel losing cell. Re-measure — do NOT assume; the resident-pool/T1 work since
  then may have shifted it either way.

### H-MARKER (the marker-port hypothesis adjacent to the disproven wreckage)
- **Predicts:** on a losing cell, a Gate-2 removal-oracle attributes a DOMINANT
  (> ~40–50%) and **non-rg-faithful** (gz does MORE marker/apply-window work than rg — a
  real structural divergence) share of the losing-cell tax to the speculative u16-marker /
  apply-window / replace-markers machinery.
- **The PORT it would justify (rg blueprint):** converge gz's marker path to rg's
  STRUCTURE where gz diverges — `deflate.hpp:1589-1666`
  (`readInternalCompressedMultiCached`, unified clean+markered read), `ChunkData::applyWindow`
  (`ChunkData.hpp:247,302`), `MarkerReplacement` / `DecodedData.hpp:305-516`. **NOT**
  deletion (rg carries the same u16 machinery: ~31.25% replaced-marker symbols, 0.113s
  applyWindow — deleting it diverges, not converges).

### PRE-REGISTERED NO-GO on the marker port (the falsifier that forbids it)
The marker-structure port is justified **ONLY** if the fresh removal-oracle shows the
marker/apply-window machinery is BOTH:
  (i) a DOMINANT (> ~40–50%) share of the losing-cell tax, AND
  (ii) a NON-rg-faithful divergence (gz does MORE than rg — measurably more marker bytes,
       more apply-window CPU, or a 2-pass where rg does 1).

**NO-GO (report the actual dominant bucket instead, do NOT start the marker port) IF ANY:**
- The tax is **DISTRIBUTED** (no single bucket > ~40–50%).
- The dominant bucket is the **SHARED PARALLEL PIPELINE** (block-finder, chunk lifecycle,
  consumer pace, tail/scheduling, effcores-low) — NOT the marker machinery.
- gz's marker path **already matches rg's structure** (gz's apply_window already ≤ rg's,
  gz marker_bytes ≈ rg's) → there is no divergence to converge, so it is not a lever.
- The dominant bucket is **clean decode / inner kernel** → that is the T1/igzip front,
  out of T≥2 scope.
- The losing cell is **silesia-T4 tail-imbalance** (few-large-late chunks, effcores ~0.92,
  in-order consumer wait) → already gated-falsified as a cheap lever (prefetch-depth A/B
  flat); report at-floor, do NOT re-walk.

### Anti-amnesia (do NOT re-bank these as fresh findings — they are CAUTIONARY priors to RE-TEST):
The banked record (`project_two_binary_matrix_2026_06_16`) already claimed, via removal-
oracle, "92% of recoverable T4 wall = marker machinery; clean decode beats rg." That was
BEFORE the view-list convergence which then hit instruction-parity and re-attributed the
residual to **tail-imbalance** (markers no longer dominant). So the "marker is the lever"
prior is STALE post-300e772b. This cycle RE-MEASURES from the current binary; the old
percentages are NOT facts and may NOT be cited as the verdict.

---

## MEASUREMENT DISCIPLINE (the gates every number must pass)

- **Gate-0:** rg-vs-rg self-test ≈ 1.0 ± spread on each box; gz-vs-gz A/A ≈ 1.0; sha==zcat
  every arm (non-inert/correct); /dev/null BOTH arms (sink law); `GZIPPY_DEBUG=1` →
  `path=ParallelSM` + build-flavor=parallel-sm+pure; built sha == requested sha (no drift,
  fresh identical builds); any oracle/knob run proven non-inert (counter fired,
  hits==expected).
- **Gate-1:** interleaved best-of-N ≥ 13 (≥7 floor); report Δ AND inter-run spread; Δ <
  spread ⇒ TIE (never a win).
- **Gate-2:** removal-oracle (region actually removed, wall measured byte-exact) is the
  ONLY verdict for attribution; slow-inject gives a SLOPE (on/off-path) NOT a prize SIZE.
- **Gate-3:** Intel (neurotic) + AMD (solvency) — a one-arch result is NOT LAW. AMD-
  specific confound: marker/window-absent path is the most likely PEXT/PDEP user →
  microcoded on Zen2 → a marker win could shrink/invert. Grep the hot path for
  `_pext_u64`/`_pdep_u64`/`bzhi` before trusting "low inversion risk."
- **Gate-4:** `path=ParallelSM` + feature fingerprint; verify the BOX binary actually
  contains HEAD (grep a symbol / confirm the built sha == requested sha — last cycle's
  push.default=tracking stale-build incident).

## BOX HYGIENE (never leave a box mutated/frozen)
- **neurotic (Intel):** LXC unfreezable (host-read-only no_turbo/governor); base-pinned
  35W part is frequency-stable → ratios trustworthy via interleave+best-of-N. Build in
  /dev/shm. No mutation.
- **solvency (AMD):** bare-metal FREEZABLE. If frozen for a LAW run: governor=performance
  + cpufreq/boost=0 (NOT intel no_turbo), idempotent, GUARANTEED restore
  (governor=ondemand/schedutil, boost=1), verified at exit + reported. NEVER leave frozen.
  Do NOT touch bench-lock.sh.

## ANTI-ORPHAN
Run synchronously in-turn; timeout long decodes; BANK after each milestone (commit+push to
`origin/kernel-converge-A`); results in `plans/T2-RG-LOCATE-RESULTS.md`. Never end on "will
resume automatically."
