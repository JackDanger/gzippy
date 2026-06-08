# DISPROOF ADVISOR VERDICT — Stage 2 copy-free-to-final WIRED + MEASURED (commit 0f5bc85b)

Two synchronous advisors this turn:
1. **Source-verify (pre-implementation)** — UPHELD the KEY REALIZATION: in the gzippy-native
   FOLD path the 32 KiB predecessor window is ALREADY the contiguous tail of chunk.data (the
   engine flips strictly BEFORE the ctx-flip, so ≥32 KiB post-flip clean bytes precede every
   back-ref), so `data_prefix_len` stays 0 and the CRC-prefix-exclusion + decode_bypass-round-trip
   landmine is ENTIRELY SIDESTEPPED. This is the faithful vendor setInitialWindow prepend, NOT the
   forbidden window-in-scratch dual-region shortcut. Named 5 hazards (H1 OOB failure-mode, H2
   stored-block, H3 commit-before-decoded_range + multi-call accounting, H4 base re-fetch after
   regrow, H5 native-only blast radius) — ALL handled in the landed code.
2. **Measurement disproof (post-implementation)** — verdict below.

## VERDICT: UPHELD-WITH-CAVEATS

The gzippy-native T8 wall MOVED FASTER from deleting the ring→chunk.data drain memcpy (Stage 2
copy-free-to-final wiring). Sign-stable, byte-exact, triangulated by the same-binary
GZIPPY_FOLD_NODRAIN knob (+0.067× last turn). Magnitude is softer than first stated.

### Per-question
- **Q1 (sign):** Sign confident but not "9/10 robust." Real sign count 8/10 (via-rg delta) or
  9/10 (head-to-head A/B); the two derivations disagree on P10 (ratio-of-bests non-transitivity).
  Per-pass Δ ≈ 1σ ⇒ a per-pass TIE; significance exists only after averaging (SE≈0.020, t≈2.9),
  and the 10 passes are AUTOCORRELATED (load drifted 2.2→5.0 monotonically) so effective N<10 ⇒
  ~2-SE confidence. The paired-delta-beats-absolute-ratio reasoning is SOUND.
- **Q2 (magnitude):** mean +0.058×, median +0.044×, SE ±0.020, CI ≈ +0.02..+0.10×. Record
  **+0.05×**; DROP the +0.07 upper edge. Load-bias direction indeterminate.
- **Q3 (A/B partner):** priornative (/tmp/gzbuild-native @9cde0b4f) is conceptually the right
  isolation (same engine, same bootstrap, differs only drain-vs-contig). Provenance VERIFIED
  this turn: 9cde0b4f→0f5bc85b production-decode delta is ONLY the Stage 2 wiring (Stage 1 is
  additive-unwired, nodrain knob OFF by default, fulcrum additive). Cross-binary layout/alignment
  is a residual confound — the CLEAN isolation is the same-binary nodrain knob, which AGREES
  (+0.067×) ⇒ good triangulation.
- **Q4 (vs +0.067× prediction):** stage2 removes strictly MORE than the nodrain knob (also
  eliminates the clean-tail ring-WRITE, which nodrain kept) but ADDS contig regrow + per-call
  window re-fetch. Measured +0.058 < +0.067 ⇒ the new contig overhead is real (~0.01×) and offsets
  the extra ring-write saving (or load-suppression). Internally consistent.
- **Q5 (byte-exact/faithfulness + next):** data_prefix_len=0 faithful-prepend UPHELD empirically
  — sha 028bd002… matches gzip AND rapidgzip on 211,968,000 bytes at T1+T8 on x86_64 AND arm64,
  plus the multi-block-clean + CRC-prefix-excluded test; a CRC-range bug would flip the sha across
  that matrix. Next bottleneck = intrinsic symbol rate (~0.11× to ocl_cf 0.925×, then
  placement/scheduler to 1.0×); drain removal doesn't touch symbol rate so +0.05× is additive.
- **Q6 (strongest disproof):** per-pass Δ≈1σ; significance rests on averaging passes whose
  independence is compromised by monotonic load drift (~2-SE); magnitude is cross-binary
  (layout-confounded); +0.07 edge exceeds mean+median. None overturns the SIGN (nodrain knob
  corroborates).

### HONEST BANKED NUMBER
native_fold ~0.74× → ~0.79× rg, **+0.05× banked** (paired mean +0.058 ±0.02 SE; median +0.044;
sign-stable 8–9/10; corroborated by same-binary GZIPPY_FOLD_NODRAIN +0.067× wrong-bytes).
Magnitude load-confounded (loadavg 2.2→5.0, autocorrelated passes ⇒ ~2-SE). Drop the +0.07 edge.

### FOLLOW-UPS (advisor)
(1) provenance git diff — DONE this turn (clean). (2) one quiet-box A/B pass to firm the magnitude
— DEFERRED (box persistently loaded 2-5; the same-binary nodrain knob already gives the clean
isolation). Sign-confident, magnitude +0.05× provisional.

## MEASUREMENT DATA (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved
## scripts/measure.sh N=11 best-of-11/pass, sha-OK every run, RAW=211968000, P=8, 10 passes)
priornative/stage2 (>1.0=stage2 faster): 0.954 0.972 0.989 0.929 [1.011] 0.876 0.805 0.969 0.818 0.969
stage2/rg:      0.800 0.786 0.741 0.722 0.765 0.789 0.880 0.853 0.819 0.709
priornative/rg: 0.763 0.765 0.733 0.671 0.773 0.692 0.725 0.741 0.689 0.731
delta(stage2-prior)/rg: +.037 +.021 +.008 +.051 -.008 +.097 +.155 +.112 +.130 -.022 (mean +.058, median +.044)
