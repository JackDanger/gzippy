# DISPROOF BRIEF — contig clean packed multi-literal fast loop (store-side technique)

## WHAT I DID
Ported the ring-path VAR_V speculative software-pipelined fast loop (igzip asm:518 —
8-byte packed multi-literal store + decode pipeline) onto the Stage-2 CONTIGUOUS
copy-free clean tail (`decode_clean_into_contig`). The contig case is simpler than the
ring (no `% U8_RING_SIZE`, no wrap-straddle). This attacks the STORE-bandwidth side the
prior advisor (Q4) named as a plausible binder: the prior contig loop wrote 2-/3-packed
literals ONE BYTE AT A TIME; the fast loop writes the packed word in one store and
advances by the leading-literal count.

## BYTE-EXACT (rule: wrong bytes = void)
gzippy-native sha 028bd002…cb410f == gzip == rapidgzip on silesia at T1+T8, guest
x86_64 AND arm64; second corpus (fold_test) matches gzip. Full lib suite 885 pass /1
pre-existing diff_ratio timing flake. Under GZIPPY_POISON_RESERVE=1: all 33
seam_crossing+3-oracle+multi-oracle+proptest tests pass; fuzz_loop_differential 5000
iters byte-exact. The poison specifically stresses the contig copy-free seam the fast
loop writes into.

## COVERAGE (fast loop is dominant, not bypassed)
careful-loop inject hits: baseline ~27M clean decode events → fastloop 8.3M ⇒ the fast
path handles ~69% of clean decode events. The wide store IS exercised.

## REMOVE-AND-MEASURE (the verdict — locked guest, interleaved measure.sh N=11, T8,
## sha-OK every run, baseline=/tmp/gz-baseline pre-fastloop vs fastloop=/tmp/gz-fastloop)
PASS 1 (load 1.40): baseline 0.1635 | fastloop 0.1679 | rg 0.1304 ⇒ fastloop 0.974× baseline
PASS 2 (load ~1.3): baseline 0.1642 | fastloop 0.1658 | rg 0.1307 ⇒ fastloop 0.990× baseline
PASS 3 (load high): baseline 0.1687 | fastloop 0.1691 | rg 0.1317 ⇒ fastloop 0.998× baseline
ALL THREE: TIE (Δ < spread, sign weakly NEGATIVE — fastloop marginally slower/equal).

## MY READ
The packed wide store (store-bandwidth side) does NOT move the wall — store bandwidth was
already slack-masked. This RESOLVES the prior advisor's Q4 in the opposite direction:
store bandwidth is NOT the binder within the loop. The residual native_fold→rg gap
(~0.77× → ocl_cf 0.925× ceiling) must live in the Huffman symbol-DECODE COMPUTE (the
lut_litlen_decode LUT lookup + bit extraction + dist_hc.decode), which the wide store
does not touch. KEEP the fast loop per rule 7a (byte-exact, faithful to ISA-L, latent
value), but it banks NO wall progress.

Reconciliation with the prior perturbation: slowing the WHOLE loop moved the wall
(on-path), but speeding ONLY the store side ties — consistent with rule 3 (slow-down
slope ≠ speed-up ceiling) AND with the store not being the loop's sub-bottleneck.

## DISPROOF QUESTIONS
1. Is the 3-pass TIE a sound basis to conclude the store side is slack, or could a
   confound (scheduling slack at 57% consumer-WAIT masking a real per-thread store win,
   load) be hiding a win? Is the next-binder-migration explanation valid?
2. Is "redirect the lever to Huffman symbol-decode COMPUTE" the correct read, or is the
   honest statement weaker (e.g. "the store technique is exhausted; the loop's binding
   sub-resource is UNIDENTIFIED — needs a decode-only perturbation/oracle to confirm it's
   the LUT decode before grinding BMI2/packed-LUT")?
3. Is keeping the fast loop on a TIE correct (rule 7a), or does the weakly-negative sign
   argue to revert?
4. Given the prompt's plateau-fork clause: is this a plateau (⇒ escalate) or just one
   technique tied (⇒ continue with the redirected decode-compute lever)?
