# DISPROOF ADVISOR VERDICT — contig clean symbol-decode causal perturbation

## VERDICT: UPHELD-WITH-CAVEATS
The perturbation cleanly establishes `decode_clean_into_contig` (the gzippy-native FOLD
production clean tail) is on the T8 critical path — monotone spin50<spin100 AND the
frequency-neutral sleep control preserves/exceeds the spin delta (rule-2 signature of
real criticality, not turbo depression). BUT it perturbs the ENTIRE loop body, so it
CANNOT attribute the residual gap specifically to Huffman symbol-decode COMPUTE vs
store/copy bandwidth. The "BMI2 / wider-store / packed-LUT will move the wall" inference
is UNPROVEN until the binding resource within the loop is isolated.

## PER-QUESTION
- Q1 (on-path): SOUND. Confound named: sleep100 > spin100 both passes ⇒ descheduling the
  worker adds wakeup-latency/consumer-starvation coupling beyond its own compute
  (consumer already 57% WAIT). Inflates the SLOPE, not the SIGN. On-path holds.
- Q2 (inject placement): RIGHT proxy for symbol-decode EVENTS (1 inject/litlen decode),
  not bytes. Under-weights backref-side (dist decode + copy get no extra inject) and
  ignores per-byte store cost. Biases slope magnitude, not the on-path verdict.
- Q3 (ceiling): ocl_cf 0.925× is correct as a LOOSE UPPER bound, correctly used instead
  of the slope (rule 3). CAVEAT: ocl_cf removes the ENTIRE engine (clean + marker +
  LUT build), so 0.925× OVER-credits what touching only decode_clean_into_contig buys.
  True clean-loop-only ceiling lies between off-rate and 0.925×; a clean-ONLY oracle is
  the owed measurement to tighten it.
- Q4 (strongest phantom, LOAD-BEARING): the inject slows the WHOLE loop body — Huffman
  decode AND packed-literal stores AND the word-copy backref AND bits.refill. A positive
  wall response proves "loop on path" but does NOT isolate that SYMBOL-DECODE COMPUTE
  (what BMI2/PEXT/packed-LUT attack) is the binding resource. On literal-heavy silesia
  the loop may be store/load-bandwidth-bound on the per-byte `base.add(*pos).write()`,
  in which case faster Huffman decode won't move the wall.

## ACTIONABLE (owner's read)
The current contig loop writes literals ONE BYTE AT A TIME even for the 2-/3-symbol
packed case (marker_inflate.rs:2096). ISA-L writes packed literals as one WIDE store.
This is the clearest ISA-L divergence and it attacks the STORE-bandwidth side the
advisor named as a plausible binder. Implement the packed multi-literal wide store
(byte-exact, vendor-faithful), remove-and-measure on the wall:
  - moves wall ⇒ banked tooth + store side was a real component.
  - ties ⇒ store side slack, lever is pure Huffman-decode rate ⇒ BMI2/packed-LUT next.
Either outcome is informative; the change is byte-exact-or-void.
