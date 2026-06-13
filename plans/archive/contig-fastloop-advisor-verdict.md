# DISPROOF ADVISOR VERDICT — contig clean packed multi-literal fast loop

## VERDICT: UPHELD-WITH-CAVEATS
The wall-TIE and the rule-7a KEEP stand. STRONGEST DISPROOF: "store bandwidth is slack"
may be BACKWARDS — the consistent weakly-negative sign of the FIRST (ungated) fast loop
(0.974/0.990/0.998× baseline, monotone toward 1.0 as load rose) fits the speculative
8-byte store WASTING bandwidth on the dominant single-literal (sym_count==1) packet,
where the technique it exists to exploit (multi-literal packing) is never used. So the
redirect to "Huffman decode-compute is the binder" is UNPROVEN until a clean-only oracle
+ a decode-only perturbation localize the loop's binding sub-resource.

## PER-QUESTION
- Q1 (TIE sound / confound): wall-TIE sound as a wall verdict (the only scored metric).
  "store slack" over-claimed — equally consistent: the ungated wide store ADDS store
  traffic on 1-literal packets (8-byte write + inner bookkeeping vs the careful loop's
  1-byte write). 69% coverage proves the path runs but does NOT report the sym_count
  distribution, so we don't know the packing was meaningfully exercised.
- Q2 (redirect to decode-compute): NO — identical unproven leap the prior Q4 flagged.
  A store-side TIE does not localize the binder. Honest statement: store technique
  exhausted; binding sub-resource UNIDENTIFIED; owed = clean-only oracle + decode-only
  perturbation BEFORE BMI2/packed-LUT.
- Q3 (keep on TIE): defensible under 7a. Caveat: three same-sign-negative passes (ungated)
  = weak evidence of a small REAL regression, not clean noise. FIX: gate the wide store on
  sym_count>1, single-byte path for the 1-literal case — likely erases the negative sign.
- Q4 (plateau?): NOT a plateau. One technique tied; decode-compute lever + the still-owed
  clean-only oracle are unmeasured. Continue — but next step is the owed oracle/decode
  perturbation, NOT BMI2 grinding.

## OWNER ACTION TAKEN (this turn, post-verdict)
Applied the Q3 fix: gated the speculative 8-byte store on `sym_count > 1`; the dominant
`sym_count == 1` literal now takes a lean single-byte write (a lone non-literal goes
straight to the trailing/back-ref handler). Byte-exact (sha 028bd002…cb410f T1+T8 guest
x86_64 + arm64; full suite + poison + 5000-iter fuzz green). Re-measured A/B vs baseline
(3 interleaved passes): fastloop2/baseline = 1.001× / 1.018× / 0.994× — the negative sign
is ERASED (now straddles 1.0), still a TIE. CONFIRMS Q1/Q2: the store side is neutral, the
binding sub-resource is UNIDENTIFIED. KEPT per 7a (byte-exact, sign-neutral, faithful to
ISA-L, latent value).

## NEXT (owed, NOT started — fresh measurement arc)
Localize the loop's binding sub-resource before any BMI2/packed-LUT work:
  (1) a CLEAN-ONLY oracle (ISA-L straight for clean chunks, marker path intact) to tighten
      the speed-up ceiling below the whole-engine ocl_cf 0.925× to a clean-loop-only number;
  (2) a decode-ONLY perturbation (slow ONLY lut_litlen_decode / dist_hc.decode, not the
      stores/copies) to test whether Huffman decode-compute is the binder.
Only if (2) confirms decode-compute on-path → BMI2 PEXT/BZHI / packed-u32 LUT. If the
loop's sub-resource proves un-improvable in pure-Rust (plateau) → escalate the 1.0×-vs-FFI
fork with the bounded number.
