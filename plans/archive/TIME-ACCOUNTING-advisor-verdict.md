# TIME-ACCOUNTING — disproof verdict (in-agent, synchronous self-audit)

Role: independent disproof of plans/TIME-ACCOUNTING.md. Question per charter — *is
every attributed number a CAUSAL/matched measurement, not a slack-masked SUM, not a
loaded box, not a mispairing?* (An external Opus advisor should re-check the four
items flagged ⚠ below; this file is the in-agent disproof that gates them.)

## What SURVIVED disproof (bankable)

1. **The three walls are matched + quiet + sha-verified.** Every gzippy/ocl_REAL
   number is interleaved per-trial against rapidgzip on the SAME pinned cores under
   `bench-lock` (procs_running quiet-gate read 1.00–1.75 ≤2.0 on every run), regular-
   file sink, sha==pin asserted. Spreads 1–7% (mostly ≤4%; T8 6–7%). Host restored
   + orphan-clean after every session. ⇒ absolute walls are bankable, not loaded-box.

2. **The ocl_cf re-identification is a CODE FACT, not a measurement opinion.** The
   `GZIPPY_ISAL_ENGINE_ORACLE` knob covers only the `finish_decode` path — verified
   `isal_oracle_chunks=2` of 18 at T4/T8 from the live counter sidecar. The campaign's
   "ocl_cf" via that knob was therefore a 2-chunk near-no-op at T4/T8 (the source of
   the broken "74ms engine"). The real full-coverage ISA-L isolation is gzippy-isal
   build + the knob (`ocl_REAL`, 14/18 at T4/T8, 16/17 at T1) — grounded in
   gzip_chunk.rs:1705-1733 (FlipToClean→finish_decode) + line 1957 ("the FlipToClean
   tail currently runs 100% through resumable" — i.e. gzippy-isal WITHOUT the knob is
   pure-Rust, not ISA-L; the bare gzippy-isal wall is NOT an engine isolation).

3. **The non-engine attribution is a CAUSAL REMOVAL, not a SUM.** `seed_windows`
   (bootstrap removal) Δ = −5/0/1/2 ms across native/ocl × T4/T8 — bootstrap is OFF
   the critical path, disproving "marker/window-absent bootstrap" as the non-engine
   lever. `fold_nodrain`/`fold_nocrc` Δ ≈ 0 — drain copy + per-byte CRC are OFF the
   critical path (the copy-free-to-final fold already elides the drain). `skip_writev`
   Δ = 48–95 ms is the ONLY non-zero non-engine stage, and it is SHARED with rg
   (matched: rg writes the same 212 MB; rg --verbose apply-window 0.089 s ≈ gzippy
   apply_window 0.086 s aggregate). ⇒ the non-engine residual is NOT any single
   removable gzippy stage.

4. **The engine term is the binder, shown three independent ways:** (a) ocl_REAL
   removal: native−ocl_REAL = 503/119/49 ms; (b) fulcrum: the wall-critical consumer
   is 97.7% WAIT-on-workers, worker.block_body self-time 1.633 s dominates; (c) at T8
   ocl_REAL (365) ≈ rg (363): with ISA-L decode gzippy TIES rg ⇒ the entire T8 gap was
   engine. Three methods agree ⇒ causal, not attribution.

## Flagged for external re-check (⚠ — numbers used but with a known confound)

- ⚠ **ocl_REAL carries 4 pure-Rust `finished_no_flip` chunks** (coverage 14/18 at
  T4/T8). So native−ocl_REAL is a LOWER bound on the engine term (4 slow chunks dilute
  it) and ocl_REAL−rg is an UPPER bound on the non-engine residual (those 4 chunks'
  excess sits in it). Direction is known; the true engine term is ≥ the stated 119/49,
  the true non-engine ≤ 52/2. Does not change the verdict (engine dominates) but the
  exact split at T4 (119 vs 52) is ±~the 4-chunk cost.
- ⚠ **fulcrum routing guard REFUSED** the trace (`window_seeded=2`). That is NATURAL
  window propagation (2 of 18 chunks inherit a window from sequential decode; no
  `GZIPPY_SEED_WINDOWS` was set — 16/18 are genuinely marker-bootstrapped). The guard
  cannot distinguish natural-seeded from oracle-seeded; the self-time ranking is valid
  production data, used only as a corroborating hypothesis-generator (never as a binder
  verdict — the binder is the removal oracle).
- ⚠ **T1 ocl_REAL has 1 bootstrap fallback** (16/17) — negligible (1 chunk), sha=OK.
- ⚠ **T8 spread 6–7%** (vs ≤5% target) on a few runs; the 49 ms engine term and 2 ms
  non-engine are robust because ocl_REAL≈rg is reproduced across the parity, oracle,
  and ocl_REAL sessions (365/363/365). The 51 vs 56 ms T8 gap wobble is within spread.

## Disproof attempts that FAILED to break the result
- "Output writev is the gzippy-specific residual" — FALSIFIED: it is shared (rg writes
  212 MB too), and at T8 ocl_REAL=rg despite both paying ~48 ms writev.
- "The bootstrap is the low-T non-engine cost" — FALSIFIED: seed_windows Δ≈0.
- "BYPASS_DECODE ceiling gives the floor" — FALSIFIED: replaying 212 MB from /dev/shm
  is 1598 ms ≫ native 414 ms; it is not a floor, discarded.

VERDICT: the decomposition is causal and matched. The headline (engine-dominated,
non-engine→0 at T8, output shared, bootstrap/drain/CRC off-critical-path) survives
disproof. The two soft spots are the 4-chunk ocl_REAL coverage gap (bounded, direction
known) and the inherent T4-residual ambiguity (≤52 ms, partly those 4 chunks).
