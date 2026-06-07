# DISPROOF-ADVISOR BRIEF — bounded pure-Rust engine ceiling (VAR_VI)

You are an INDEPENDENT, READ-ONLY disproof advisor. Adversarial intent: try to BREAK
the PLATEAU verdict below. Source-verify any file:line you rely on. Do NOT trust this
brief's conclusions — re-derive them.

## THE QUESTION
Can a FAITHFUL-u8 pure-Rust DEFLATE engine reach igzip-class (≥~0.85× ISA-L) in
isolation, with inline-ASM-equivalent techniques? This bounds whether the gzippy→1.0×
parity bar is reachable WITHOUT C-FFI. The old "2.4× plateau = unreachable" was measured
on the DISCREDITED u16-ring arch and does NOT bound the current faithful-u8 engine.

## WHAT WAS BUILT THIS TURN
`benches/engine_isolation.rs` VAR_VI (`decode_var_vi`, x86_64) = VAR_V (committed faithful-u8
speculative software-pipelined flat-u8 loop + igzip packed-u32 multi-symbol table reuse,
trick #1+#2+#3) PLUS the two remaining igzip techniques the kernel bench had not measured:
  (1) BMI2 BZHI (`_bzhi_u64`, `bzhi64`) for the variable-width distance extra-bits extraction
      (replaces the materialized `(1<<extra)-1` mask in the hot path);
  (2) AVX2/SSE MOVDQU wide overlap-copy back-ref (`avx_backref_copy`): 32-byte AVX2 stores
      for the non-overlapping bulk, 16-byte SSE for distance>=16 overlap, RLE memset for
      distance==1, byte tail otherwise. igzip uses SSE xmm MOVDQU (asm:603-627); generalized
      to AVX2 width here.
trick #3 (packed-u32 short table) is CONFIRMED fully exploited: VAR_VI drives the SAME
`LutLitLenCode::decode` packed packets and unpacks up to 3 packed literals per decode
(the multi-symbol short-code table, lut_huffman.rs:1021-1063).

## PRE-REGISTERED FALSIFIER
- PASS (≈igzip-class): VAR_VI/(III isal) ≥ ~0.85 in isolation ⇒ pure-Rust CAN be igzip-class.
- PLATEAU: VAR_VI stays well below ~0.85 (e.g. ~0.55-0.7) WITH all igzip techniques ⇒
  pure-Rust igzip-class likely UNREACHABLE ⇒ the 1.0×-vs-no-FFI fork is REAL & HARD-BOUNDED.

## MEASUREMENT (locked guest REDACTED_IP via double-ssh; 16c gov=performance; load ~3.3;
   taskset -c 0; N=11 interleaved; native target-cpu = BMI2+AVX2 LIVE, avx2_detected=true;
   2 independent runs, STABLE)
Per-chunk VAR_VI vs ISA-L (vs_iii), run1/run2:
  67145126 : 0.640 / 0.596    167879427: 0.586
  268518043: 0.562 / 0.568    369261644: 0.560 / 0.552    469882459: 0.590 / 0.595
AGGREGATE (median-of-per-chunk-medians, MB/s): ISA-L 847-851; VAR_V 460-462; VAR_VI 504-525.
  ⇒ VAR_VI aggregate ≈ 0.59-0.62× ISA-L; per-chunk 0.55-0.64×.
  ⇒ BMI2+AVX added ~9-14% over VAR_V (0.54→0.60×) but did NOT close the gap.
SELFTEST=PASS (median-chunk iii/i=2.73, in [2.5,3.6]).

## BYTE-EXACTNESS (the load-bearing correctness claim)
VAR_VI printed an MBps number on EVERY chunk (never "VOID"). The bench prints MBps only when
`exact[k]==true` (engine_isolation.rs:802), and `exact[k] = o.len()>=n_actual && o[..n]==scalar
&& scalar_eq_isal` (:744). So VAR_VI matched the SCALAR reference AND scalar matched ISA-L on
every swept chunk ⇒ VAR_VI is byte-exact vs BOTH scalar and ISA-L.
NOTE: the bench's top-line `SHA_ALL_EQUAL=no` is from the PRE-EXISTING VAR_IV_E000/E2/E23/E234
failures (a separate read_clean_e234 path, NOT touched this turn) — it does NOT reflect on VAR_VI.

## VERDICT CLAIMED: PLATEAU
VAR_VI ≈ 0.59-0.62× ISA-L (per-chunk 0.55-0.64), ~23pp below the 0.85 PASS line, WITH the full
igzip technique stack (packed-u32 table + speculative 8B store + preload pipeline + slop headroom
+ BMI2 BZHI + AVX2 MOVDQU overlap copy) on the faithful flat-u8 buffer. ⇒ pure-Rust igzip-class
is likely UNREACHABLE on this engine; the 1.0×-vs-no-FFI user-constraint fork is REAL and now
HARD-BOUNDED at ~0.6× ISA-L per-thread engine throughput.

## YOUR JOB — try to BREAK this:
1. Is VAR_VI a FAITHFUL representation of "best pure-Rust+ASM" or is a technique missing/crippled?
   (e.g. is BZHI actually emitted? does AVX2 copy actually run? is the speculative store real?
   is the careful tail dominating? is the header/table-build folded into the timed region inflating
   the denominator unfairly — note ISA-L (iii) decodes the SAME slice incl. its own header parse,
   so it's symmetric.)
2. Is the byte-exactness reasoning airtight (does printing MBps truly prove exact[k]==true)?
3. Is the 0.85 bar the right PASS line, or could a real production integration of a ~0.6×
   isolation engine still TIE the 1.0× WALL (i.e. is isolation the wrong instrument for the fork)?
4. Any confound in the ratio (load 3.3, turbo on, core-0 pin, Vec alloc per call, return-to-Vec
   copy charged to all variants equally)?

Output a verdict: PLATEAU UPHELD / UPHELD-WITH-CAVEATS / REFUTED, with the load-bearing reasons.
Write it to plans/engine-ceiling-advisor-verdict.md.
