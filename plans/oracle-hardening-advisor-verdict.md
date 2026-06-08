# oracle-hardening-advisor-verdict.md — independent disproof of the hardened oracle + re-measured ocl_cf

**Advisor:** independent disproof (synchronous `claude -p --model opus`).
**Date:** 2026-06-08. **Subject:** the TOOLING-AUDIT-driven oracle hardening + reserve-confound
fix (commit e4389f05 + factor-8 refinement) and the re-measured ocl_cf / native low-T cells.

## VERDICT: UPHELD-WITH-CAVEATS

### Break attempts + what survived
1. **Reserve-fix byte-exactness — VERIFIED SAFE.** The change is wholly inside
   `finish_decode_chunk_isal_oracle`, gated by `GZIPPY_ISAL_ENGINE_ORACLE`; production native
   path untouched ⇒ byte-exact. Factor-8 under-reserve CANNOT silently contaminate: the FFI sets
   `avail_out=out.len()` (no OOB), returns `None` on overflow → `Ok(false)` → pure-Rust fallback
   → counter++ → in-script assert VOIDs. A truncated `Some` is also blocked (boundary/end check).
   CAVEAT: factor 8 is only ~2.4× over silesia's file-average 3.3×; a locally-compressible chunk
   can exceed 8× and VOID the run — LOUD not silent, but expect occasional VOIDs.

2. **ocl_cf ratio under load — direction UPHELD, magnitude SOFT.** Interleaved A/B cancels
   common-mode load, so "0.945 was PESSIMISTIC, true ceiling ≈ TIE" survives. BUT 0.997 ± 12%
   ⇒ lower bound 0.877 < 0.945; the SINGLE run does not itself clear 0.945 — only the 3-run
   0.98–1.00 cluster does. **BANK "≈TIE / ≥0.945×", NOT "0.997×".**

3. **Engine-bound CLOSABLE (the A-vs-B fork input) — PARTIALLY REFUTED.**
   - LOSS EXISTS: YES, strong. T1 native 0.608× (4% spread) == isolated inner-loop 0.55–0.60×
     ⇒ genuinely engine-bound.
   - CLOSABLE-by-a-better-engine: NOT established at these spreads. T4 native 0.755× (3%) vs
     ocl_cf 0.900× (**16%**) — ocl_cf's lower bound 0.900−0.144 ≈ 0.756 ≈ native ⇒ within spread
     the engine swap does NOTHING and the residual is non-engine (placement/scheduling). The wide
     ocl_cf-T4 spread ADMITS THE NULL. **Tighten ocl_cf T4 to ≤5% spread before banking
     "⅔ of the gap is engine."**
   - T1 ocl_cf is VOID (bootstrap window-absent fallback — legitimate property) ⇒ T1 gives an
     engine-bound LOSS but NO ISA-L ceiling, so it cannot itself prove closability. Don't cite T1
     for "ISA-L closes the gap."

### Net (advisor's words)
The fork (full-kernel asm) has a REAL engine-bound loss to chase, but "substantially engine-rate
/ ⅔ closable" is NOT yet measurement-clean. The OWED next datum to make option-B clean: a
TIGHT (≤5% spread, genuinely quiet box) ocl_cf-T4 vs native-T4 vs rg, to test whether the engine
swap actually closes the T4 gap or the residual is non-engine.
